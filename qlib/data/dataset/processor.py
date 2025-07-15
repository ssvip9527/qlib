# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, Text, Optional
import numpy as np
import pandas as pd

from qlib.utils.data import robust_zscore, zscore
from ...constant import EPS
from .utils import fetch_df_by_index
from ...utils.serial import Serializable
from ...utils.paral import datetime_groupby_apply
from qlib.data.inst_processor import InstProcessor
from qlib.data import D


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    从多级索引列DataFrame中获取一组列

    参数
    ----------
    df : pd.DataFrame
        具有多级列的DataFrame。
    group : str
        特征组的名称，即组索引的第一级值。
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class Processor(Serializable):
    def fit(self, df: pd.DataFrame = None):
        """
        学习数据处理参数

        参数
        ----------
        df : pd.DataFrame
            当我们使用处理器逐个拟合和处理数据时，fit函数依赖于前一个处理器的输出，即`df`。

        """

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        处理数据

        注意：**处理器可能会就地修改`df`的内容！！！**
        用户应在外部保留数据的副本

        参数
        ----------
        df : pd.DataFrame
            处理器的原始数据或前一个处理器的结果。
        """

    def is_for_infer(self) -> bool:
        """
        此处理器是否可用于推理
        某些处理器不可用于推理。

        返回
        -------
        bool:
            是否可用于推理。
        """
        return True

    def readonly(self) -> bool:
        """
        处理器在处理时是否将输入数据视为只读（即不写入输入数据）

        了解只读信息有助于处理器避免不必要的复制
        """
        return False

    def config(self, **kwargs):
        attr_list = {"fit_start_time", "fit_end_time"}
        for k, v in kwargs.items():
            if k in attr_list and hasattr(self, k):
                setattr(self, k, v)

        for attr in attr_list:
            if attr in kwargs:
                kwargs.pop(attr)
        super().config(**kwargs)


class DropnaProcessor(Processor):
    def __init__(self, fields_group=None):
        """
        参数
        ----------
        fields_group :
            要处理的字段组名称。
        """
        self.fields_group = fields_group

    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))

    def readonly(self):
        return True


class DropnaLabel(DropnaProcessor):
    def __init__(self, fields_group="label"):
        """
        参数
        ----------
        fields_group : str, 默认"label"
            标签字段组名称。
        """
        super().__init__(fields_group=fields_group)

    def is_for_infer(self) -> bool:
        """根据标签删除样本，因此不可用于推理"""
        return False


class DropCol(Processor):
    def __init__(self, col_list=[]):
        """
        参数
        ----------
        col_list : list, 默认[]
            要删除的列名列表。
        """
        self.col_list = col_list

    def __call__(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            mask = df.columns.get_level_values(-1).isin(self.col_list)
        else:
            mask = df.columns.isin(self.col_list)
        return df.loc[:, ~mask]

    def readonly(self):
        return True


class FilterCol(Processor):
    def __init__(self, fields_group="feature", col_list=[]):
        """
        参数
        ----------
        fields_group : str, 默认"feature"
            要筛选的字段组名称。
        col_list : list, 默认[]
            要保留的列名列表。
        """
        self.fields_group = fields_group
        self.col_list = col_list

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        all_cols = df.columns
        diff_cols = np.setdiff1d(all_cols.get_level_values(-1), cols.get_level_values(-1))
        self.col_list = np.union1d(diff_cols, self.col_list)
        mask = df.columns.get_level_values(-1).isin(self.col_list)
        return df.loc[:, mask]

    def readonly(self):
        return True


class TanhProcess(Processor):
    """使用tanh处理噪声数据"""

    def __call__(self, df):
        def tanh_denoise(data):
            mask = data.columns.get_level_values(1).str.contains("LABEL")
            col = df.columns[~mask]
            data[col] = data[col] - 1
            data[col] = np.tanh(data[col])

            return data

        return tanh_denoise(df)


class ProcessInf(Processor):
    """处理无穷大值"""

    def __call__(self, df):
        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    # FIXME: 这种行为非常奇怪
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = datetime_groupby_apply(data, process_inf)
            data.sort_index(inplace=True)
            return data

        return replace_inf(df)


class Fillna(Processor):
    """处理NaN值"""

    def __init__(self, fields_group=None, fill_value=0):
        """
        参数
        ----------
        fields_group :
            要填充的字段组名称，为None时填充所有字段。
        fill_value : int, 默认0
            用于填充NaN的值。
        """
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            # this implementation is extremely slow
            # df.fillna({col: self.fill_value for col in cols}, inplace=True)
            df[self.fields_group] = df[self.fields_group].fillna(self.fill_value)
        return df


class MinMaxNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        """
        参数
        ----------
        fit_start_time :
            拟合数据的开始时间。
        fit_end_time :
            拟合数据的结束时间。
        fields_group :
            要归一化的字段组名称。
        """
        # 注意：正确设置`fit_start_time`和`fit_end_time`非常重要！！！
        # `fit_end_time`**绝对不能**包含测试数据的任何信息！！！
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.min_val = np.nanmin(df[cols].values, axis=0)
        self.max_val = np.nanmax(df[cols].values, axis=0)
        self.ignore = self.min_val == self.max_val
        # To improve the speed, we set the value of `min_val` to `0` for the columns that do not need to be processed,
        # and the value of `max_val` to `1`, when using `(x - min_val) / (max_val - min_val)` for uniform calculation,
        # the columns that do not need to be processed will be calculated by `(x - 0) / (1 - 0)`,
        # as you can see, the columns that do not need to be processed, will not be affected.
        for _i, _con in enumerate(self.ignore):
            if _con:
                self.min_val[_i] = 0
                self.max_val[_i] = 1
        self.cols = cols

    def __call__(self, df):
        def normalize(x, min_val=self.min_val, max_val=self.max_val):
            return (x - min_val) / (max_val - min_val)

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class ZScoreNorm(Processor):
    """ZScore Normalization"""

    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        # To improve the speed, we set the value of `std_train` to `1` for the columns that do not need to be processed,
        # and the value of `mean_train` to `0`, when using `(x - mean_train) / std_train` for uniform calculation,
        # the columns that do not need to be processed will be calculated by `(x - 0) / 1`,
        # as you can see, the columns that do not need to be processed, will not be affected.
        for _i, _con in enumerate(self.ignore):
            if _con:
                self.std_train[_i] = 1
                self.mean_train[_i] = 0
        self.cols = cols

    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train):
            return (x - mean_train) / std_train

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class RobustZScoreNorm(Processor):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        self.cols = get_group_columns(df, self.fields_group)
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += EPS
        self.std_train *= 1.4826

    def __call__(self, df):
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train
        if self.clip_outlier:
            X = np.clip(X, -3, 3)
        df[self.cols] = X
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore"):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __call__(self, df):
        # try not modify original dataframe
        if not isinstance(self.fields_group, list):
            self.fields_group = [self.fields_group]
        # depress warning by references:
        # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#getting-and-setting-options
        with pd.option_context("mode.chained_assignment", None):
            for g in self.fields_group:
                cols = get_group_columns(df, g)
                df[cols] = df[cols].groupby("datetime", group_keys=False).apply(self.zscore_func)
        return df


class CSRankNorm(Processor):
    """
    Cross Sectional Rank Normalization.
    "Cross Sectional" is often used to describe data operations.
    The operations across different stocks are often called Cross Sectional Operation.

    For example, CSRankNorm is an operation that grouping the data by each day and rank `across` all the stocks in each day.

    Explanation about 3.46 & 0.5

    .. code-block:: python

        import numpy as np
        import pandas as pd
        x = np.random.random(10000)  # for any variable
        x_rank = pd.Series(x).rank(pct=True)  # if it is converted to rank, it will be a uniform distributed
        x_rank_norm = (x_rank - x_rank.mean()) / x_rank.std()  # Normally, we will normalize it to make it like normal distribution

        x_rank.mean()   # accounts for 0.5
        1 / x_rank.std()  # accounts for 3.46

    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        t = df[cols].groupby("datetime", group_keys=False).rank(pct=True)
        t -= 0.5
        t *= 3.46  # NOTE: towards unit std
        df[cols] = t
        return df


class CSZFillna(Processor):
    """Cross Sectional Fill Nan"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime", group_keys=False).apply(lambda x: x.fillna(x.mean()))
        return df


class HashStockFormat(Processor):
    """Process the storage of from df into hasing stock format"""

    def __call__(self, df: pd.DataFrame):
        from .storage import HashingStockStorage  # pylint: disable=C0415

        return HashingStockStorage.from_df(df)


class TimeRangeFlt(InstProcessor):
    """
    This is a filter to filter stock.
    Only keep the data that exist from start_time to end_time (the existence in the middle is not checked.)
    WARNING:  It may induce leakage!!!
    """

    def __init__(
        self,
        start_time: Optional[Union[pd.Timestamp, str]] = None,
        end_time: Optional[Union[pd.Timestamp, str]] = None,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        start_time : Optional[Union[pd.Timestamp, str]]
            The data must start earlier (or equal) than `start_time`
            None indicates data will not be filtered based on `start_time`
        end_time : Optional[Union[pd.Timestamp, str]]
            similar to start_time
        freq : str
            The frequency of the calendar
        """
        # Align to calendar before filtering
        cal = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
        self.start_time = None if start_time is None else cal[0]
        self.end_time = None if end_time is None else cal[-1]

    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        if (
            df.empty
            or (self.start_time is None or df.index.min() <= self.start_time)
            and (self.end_time is None or df.index.max() >= self.end_time)
        ):
            return df
        return df.head(0)
