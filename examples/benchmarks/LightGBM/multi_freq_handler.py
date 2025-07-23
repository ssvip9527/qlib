#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pandas as pd

from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.data.handler import DataHandlerLP, _DEFAULT_LEARN_PROCESSORS, check_transform_proc


class Avg15minLoader(QlibDataLoader):
    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df = super(Avg15minLoader, self).load(instruments, start_time, end_time)
        if self.is_group:
            # feature_day（日频）和feature_15min（1分钟频率，每15分钟平均）重命名为feature
            df.columns = df.columns.map(lambda x: ("feature", x[1]) if x[0].startswith("feature") else x)
        return df


class Avg15minHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        data_loader = Avg15minLoader(
            config=self.loader_config(), filter_pipe=filter_pipe, freq=freq, inst_processors=inst_processors
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def loader_config(self):
        # 数据集结果：df: pd.DataFrame
        #   len(df.columns) == 6 + 6 * 16, len(df.index.get_level_values(level="datetime").unique()) == T
        #   df.columns: close0, close1, ..., close16, open0, ..., open16, ..., vwap16
        #       freq == day:
        #           close0, open0, low0, high0, volume0, vwap0
        #       freq == 1min:
        #           close1, ..., close16, ..., vwap1, ..., vwap16
        #   df.index.name == ["datetime", "instrument"]: pd.MultiIndex
        # 示例：
        #                          feature                        ...                  label
        #                           close0      open0       low0  ... vwap1 vwap16    LABEL0
        # datetime   instrument                                   ...
        # 2020-10-09 SH600000    11.794546  11.819587  11.769505  ...   NaN    NaN -0.005214
        # 2020-10-15 SH600000    12.044961  11.944795  11.932274  ...   NaN    NaN -0.007202
        # ...                          ...        ...        ...  ...   ...    ...       ...
        # 2021-05-28 SZ300676     6.369684   6.495406   6.306568  ...   NaN    NaN -0.001321
        # 2021-05-31 SZ300676     6.601626   6.465643   6.465130  ...   NaN    NaN -0.023428

        # 日频特征：len(columns) == 6, freq = day
        # $close是当前交易日的收盘价：
        #   如果用户需要获取最近T天前的`close`，使用Ref($close, T-1)，例如：
        #                                    $close  Ref($close, 1)  Ref($close, 2)  Ref($close, 3)  Ref($close, 4)
        #         instrument datetime
        #         SH600519   2021-06-01  244.271530
        #                    2021-06-02  242.205917      244.271530
        #                    2021-06-03  242.229889      242.205917      244.271530
        #                    2021-06-04  245.421524      242.229889      242.205917      244.271530
        #                    2021-06-07  247.547089      245.421524      242.229889      242.205917      244.271530

        # 警告：Ref($close, N)，如果N == 0，Ref($close, N) ==> $close

        fields = ["$close", "$open", "$low", "$high", "$volume", "$vwap"]
        # 名称：close0, open0, ..., vwap0
        names = list(map(lambda x: x.strip("$") + "0", fields))

        config = {"feature_day": (fields, names)}

        # 15分钟特征：len(columns) == 6 * 16, freq = 1min
        #   $close是当前交易日的收盘价：
        #       如果用户获取最近T天的第i个15分钟的'close'，使用`Ref(Mean($close, 15), (T-1) * 240 + i * 15)`，例如：
        #                                    Ref(Mean($close, 15), 225)  Ref(Mean($close, 15), 465)  Ref(Mean($close, 15), 705)
        #             instrument datetime
        #             SH600519   2021-05-31                  241.769897                  243.077942                  244.712997
        #                        2021-06-01                  244.271530                  241.769897                  243.077942
        #                        2021-06-02                  242.205917                  244.271530                  241.769897

        # 警告：Ref(Mean($close, 15), N)，如果N == 0，Ref(Mean($close, 15), N) ==> Mean($close, 15)

        # 当前脚本的结果：
        #   时间：  09:00 --> 09:14,            ..., 14:45 --> 14:59
        #   字段：  Ref(Mean($close, 15), 225), ..., Mean($close, 15)
        #   名称：  close1,                     ..., close16
        #

        # 表达式说明：以close为例
        #   Mean($close, 15) ==> df["$close"].rolling(15, min_periods=1).mean()
        #   Ref(Mean($close, 15), 15) ==> df["$close"].rolling(15, min_periods=1).mean().shift(15)

        #   注意：每个交易日的最后数据是第i个15分钟的平均值

        # 平均值：
        #   每个交易日第i个15分钟周期的平均值：1 <= i <= 250 // 16
        #       平均值(15分钟)：Ref(Mean($close, 15), 240 - i * 15)
        #
        #   每个交易日前15分钟的平均值；i = 1
        #       平均值(09:00 --> 09:14), df.index.loc["09:14"]: Ref(Mean($close, 15), 240- 1 * 15) ==> Ref(Mean($close, 15), 225)
        #   每个交易日最后15分钟的平均值；i = 16
        #       平均值(14:45 --> 14:59), df.index.loc["14:59"]: Ref(Mean($close, 15), 240 - 16 * 15) ==> Ref(Mean($close, 15), 0) ==> Mean($close, 15)

        # 15分钟重采样到日频
        #   df.resample("1d").last()
        tmp_fields = []
        tmp_names = []
        for i, _f in enumerate(fields):
            _fields = [f"Ref(Mean({_f}, 15), {j * 15})" for j in range(1, 240 // 15)]
            _names = [f"{names[i][:-1]}{int(names[i][-1])+j}" for j in range(240 // 15 - 1, 0, -1)]
            _fields.append(f"Mean({_f}, 15)")
            _names.append(f"{names[i][:-1]}{int(names[i][-1])+240 // 15}")
            tmp_fields += _fields
            tmp_names += _names
        config["feature_15min"] = (tmp_fields, tmp_names)
        # label
        config["label"] = (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
        return config
