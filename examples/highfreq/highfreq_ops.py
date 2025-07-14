import numpy as np
import pandas as pd
import importlib
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.contrib.ops.high_freq import get_calendar_day


class DayLast(ElemOperator):
    """DayLast操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        每个值等于其所在交易日最后一个值的序列
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
        前向填充缺失值的特征
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.ffill()


class BFillNan(ElemOperator):
    """BFillNan操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        后向填充缺失值的特征
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.bfill()


class Date(ElemOperator):
    """Date操作符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        每个值为特征索引对应日期的序列
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
        满足条件(feature_left)的特征值(feature_right)

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
        指示特征是否为缺失值的序列
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


class Cut(ElemOperator):
    """Cut操作符

    参数
    ----------
    feature : Expression
        特征实例
    l : int
        l > 0，删除特征的前l个元素（默认为None，表示0）
    r : int
        r < 0，删除特征的后-r个元素（默认为None，表示0）
    返回
    ----------
    feature:
        从特征中删除前l个和后-r个元素后的序列。
        注意：是从原始数据中删除，而不是从切片数据中删除
    """

    def __init__(self, feature, l=None, r=None):
        self.l = l
        self.r = r
        if (self.l is not None and self.l <= 0) or (self.r is not None and self.r >= 0):
            raise ValueError("Cut operator l should > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.l : self.r]

    def get_extended_window_size(self):
        ll = 0 if self.l is None else self.l
        rr = 0 if self.r is None else abs(self.r)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
