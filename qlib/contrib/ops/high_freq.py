# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import pandas as pd
from datetime import datetime

from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.data.ops import ElemOperator, PairOperator
from qlib.utils.time import time_to_day_index


def get_calendar_day(freq="1min", future=False):
    """
    使用内存缓存加载高频日历日期。
    !!!注意: 加载日历相当慢。因此在开始多进程之前加载日历会使其更快。

    参数
    ----------
    freq : str
        读取日历文件的频率。
    future : bool
        是否包含未来交易日。

    返回
    -------
    _calendar:
        日期数组。
    """
    flag = f"{freq}_future_{future}_day"
    if flag in H["c"]:
        _calendar = H["c"][flag]
    else:
        _calendar = np.array(list(map(lambda x: x.date(), Cal.load_calendar(freq, future))))
        H["c"][flag] = _calendar
    return _calendar


def get_calendar_minute(freq="day", future=False):
    """使用内存缓存加载高频日历分钟数据"""
    flag = f"{freq}_future_{future}_day"
    if flag in H["c"]:
        _calendar = H["c"][flag]
    else:
        _calendar = np.array(list(map(lambda x: x.minute // 30, Cal.load_calendar(freq, future))))
        H["c"][flag] = _calendar
    return _calendar


class DayCumsum(ElemOperator):
    """DayCumsum操作符，在开始时间和结束时间期间计算累积和。

    参数
    ----------
    feature : Expression
        特征实例
    start : str
        日内回测的开始时间。
        !!!注意: "9:30"表示时间段(9:30, 9:31)在交易中。
    end : str
        日内回测的结束时间。
        !!!注意: "14:59"表示时间段(14:59, 15:00)在交易中，
                但(15:00, 15:01)不在交易中。
        因此start="9:30"和end="14:59"表示全天交易。

    返回
    ----------
    feature:
        一个序列，其中每个值等于开始时间和结束时间期间的累积和值。
        否则，值为零。
    """

    def __init__(self, feature, start: str = "9:30", end: str = "14:59", data_granularity: int = 1):
        self.feature = feature
        self.start = datetime.strptime(start, "%H:%M")
        self.end = datetime.strptime(end, "%H:%M")

        self.morning_open = datetime.strptime("9:30", "%H:%M")  # 上午开盘时间
        self.morning_close = datetime.strptime("11:30", "%H:%M")  # 上午收盘时间
        self.noon_open = datetime.strptime("13:00", "%H:%M")  # 下午开盘时间
        self.noon_close = datetime.strptime("15:00", "%H:%M")  # 下午收盘时间

        self.data_granularity = data_granularity
        self.start_id = time_to_day_index(self.start) // self.data_granularity
        self.end_id = time_to_day_index(self.end) // self.data_granularity
        assert 240 % self.data_granularity == 0

    def period_cusum(self, df):
        """计算指定时间段内的累积和

        参数
        ----------
        df : pd.DataFrame
            输入数据

        返回
        ----------
        df : pd.DataFrame
            处理后的累积和数据
        """
        df = df.copy()
        assert len(df) == 240 // self.data_granularity
        df.iloc[0 : self.start_id] = 0  # 开始时间前的值设为0
        df = df.cumsum()  # 计算累积和
        df.iloc[self.end_id + 1 : 240 // self.data_granularity] = 0  # 结束时间后的值设为0
        return df

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform(self.period_cusum)


class DayLast(ElemOperator):
    """DayLast操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个序列，其中每个值等于其所在日的最后一个值
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform("last")


class FFillNan(ElemOperator):
    """FFillNan操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        前向填充NaN的特征
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="ffill")


class BFillNan(ElemOperator):
    """BFillNan操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        后向填充NaN的特征
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="bfill")


class Date(ElemOperator):
    """Date操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个序列，其中每个值是与feature.index对应的日期
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    """Select操作符

    参数
    ----------
    feature_left : Expression
        特征实例，选择条件
    feature_right : Expression
        特征实例，选择值

    返回
    ----------
    feature:
        满足条件(feature_left)的value(feature_right)

    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(instrument, start_index, end_index, freq)
        series_feature = self.feature_right.load(instrument, start_index, end_index, freq)
        return series_feature.loc[series_condition]


class IsNull(ElemOperator):
    """IsNull操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        指示特征是否为nan的序列
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


class IsInf(ElemOperator):
    """IsInf操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        指示特征是否为inf的序列
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return np.isinf(series)


class Cut(ElemOperator):
    """Cut操作符

    参数
    ----------
    feature : Expression
        特征实例
    left : int
        left > 0，删除特征的前left个元素（默认为None，表示0）
    right : int
        right < 0，删除特征的后-right个元素（默认为None，表示0）
    返回
    ----------
    feature:
        从特征中删除前left个和后-right个元素的序列。
        注意：是从原始数据中删除，而不是从切片数据中
    """

    def __init__(self, feature, left=None, right=None):
        self.left = left
        self.right = right
        if (self.left is not None and self.left <= 0) or (self.right is not None and self.right >= 0):
            raise ValueError("Cut operator l shoud > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.left : self.right]

    def get_extended_window_size(self):
        ll = 0 if self.left is None else self.left
        rr = 0 if self.right is None else abs(self.right)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
