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
            # 这种实现方式非常慢
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
        # 为了提高速度，对于不需要处理的列，我们将`min_val`设为`0`，
        # 将`max_val`设为`1`，这样在使用`(x - min_val) / (max_val - min_val)`统一计算时，
        # 不需要处理的列将被计算为`(x - 0) / (1 - 0)`，
        # 可以看到，不需要处理的列不会受到影响。
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
    """ZScore标准化

    对数据进行ZScore标准化处理：(x - mean) / std
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # 注意：正确设置`fit_start_time`和`fit_end_time`非常重要！！！
        # `fit_end_time`绝对不能包含测试数据的任何信息！！！
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        # 为了提高速度，对于不需要处理的列，我们将`std_train`设为1，`mean_train`设为0
        # 这样在使用`(x - mean_train) / std_train`统一计算时
        # 不需要处理的列将被计算为`(x - 0) / 1`
        # 可以看到，不需要处理的列不会受到影响
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
    """鲁棒ZScore标准化

    使用鲁棒统计量进行ZScore标准化：
        均值(x) = 中位数(x)
        标准差(x) = 中位数绝对偏差(MAD) * 1.4826

    参考:
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        # 注意：正确设置`fit_start_time`和`fit_end_time`非常重要！！！
        # `fit_end_time`绝对不能包含测试数据的任何信息！！！
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
    """横截面ZScore标准化

    对每个时间点的横截面数据进行ZScore标准化处理
    """

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
    """横截面排名标准化

    "横截面"通常用于描述数据操作。
    对不同股票的操作通常称为横截面操作。

    例如，CSRankNorm是按天分组并对每天的所有股票进行排名操作。

    关于3.46和0.5的解释：

    .. code-block:: python

        import numpy as np
        import pandas as pd
        x = np.random.random(10000)  # 任意变量
        x_rank = pd.Series(x).rank(pct=True)  # 转换为排名后将是均匀分布
        x_rank_norm = (x_rank - x_rank.mean()) / x_rank.std()  # 通常我们会将其标准化为正态分布

        x_rank.mean()   # 对应0.5
        1 / x_rank.std()  # 对应3.46
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
    """横截面填充缺失值

    对每个时间点的横截面数据填充缺失值为该时间点的均值
    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime", group_keys=False).apply(lambda x: x.fillna(x.mean()))
        return df


class HashStockFormat(Processor):
    """将数据框处理为哈希股票存储格式

    将DataFrame转换为HashingStockStorage格式
    """

    def __call__(self, df: pd.DataFrame):
        from .storage import HashingStockStorage  # pylint: disable=C0415

        return HashingStockStorage.from_df(df)


class TimeRangeFlt(InstProcessor):
    """股票时间范围过滤器

    只保留从start_time到end_time存在的数据(不检查中间是否存在)
    警告：可能导致数据泄露！！！
    """

    def __init__(
        self,
        start_time: Optional[Union[pd.Timestamp, str]] = None,
        end_time: Optional[Union[pd.Timestamp, str]] = None,
        freq: str = "day",
    ):
        """
        参数
        ----------
        start_time : Optional[Union[pd.Timestamp, str]]
            数据必须早于(或等于)`start_time`开始
            None表示不基于`start_time`过滤数据
        end_time : Optional[Union[pd.Timestamp, str]]
            类似于start_time
        freq : str
            日历频率
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
