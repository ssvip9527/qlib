# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Qlib遵循以下逻辑支持时点数据库

对于每个股票，其数据格式为<观察时间, 特征>。表达式引擎支持对此类格式数据进行计算

要计算特定观察时间t的特征值f_t，将使用格式为<周期时间, 特征>的数据。
例如，在20190719（观察时间）计算过去4个季度（周期时间）的平均收益

<周期时间, 特征>和<观察时间, 特征>数据的计算均依赖表达式引擎，分为两个阶段：
1) 在每个观察时间t计算<周期时间, 特征>，并将其合并为一个点（类似于普通特征）
2) 拼接所有合并后的数据，得到格式为<观察时间, 特征>的数据。
Qlib将使用运算符`P`执行合并操作。
"""
import numpy as np
import pandas as pd
from qlib.data.ops import ElemOperator
from qlib.log import get_module_logger
from .data import Cal


class P(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        resample_data = np.empty(end_index - start_index + 1, dtype="float32")

        for cur_index in range(start_index, end_index + 1):
            cur_time = _calendar[cur_index]
            # 为准确加载表达式，需要更多的历史数据
            start_ws, end_ws = self.feature.get_extended_window_size()
            if end_ws > 0:
                raise ValueError(
                    "时点数据库不支持引用未来周期（例如不支持`Ref('$$roewa_q', -1)`这样的表达式"
                )

            # 计算值始终是最后一个元素，因此结束偏移量为零。
            try:
                s = self._load_feature(instrument, -start_ws, 0, cur_time)
                resample_data[cur_index - start_index] = s.iloc[-1] if len(s) > 0 else np.nan
            except FileNotFoundError:
                get_module_logger("base").warning(f"警告：未找到{str(self)}的周期数据")
                return pd.Series(dtype="float32", name=str(self))

        resample_series = pd.Series(
            resample_data, index=pd.RangeIndex(start_index, end_index + 1), dtype="float32", name=str(self)
        )
        return resample_series

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time)

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0


class PRef(P):
    def __init__(self, feature, period):
        super().__init__(feature)
        self.period = period

    def __str__(self):
        return f"{super().__str__()}[{self.period}]"

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time, self.period)
