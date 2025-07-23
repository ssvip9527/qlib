# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import inspect
import logging
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Union, cast

import numpy as np
import pandas as pd

import qlib.utils.index_data as idd

from ..log import get_module_logger
from ..utils.index_data import IndexData, SingleData
from ..utils.resam import resam_ts_data, ts_data_last
from ..utils.time import Freq, is_single_value


class BaseQuote:
    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        self.logger = get_module_logger("online operator", level=logging.INFO)

    def get_all_stock(self) -> Iterable:
        """return all stock codes

        Return
        ------
        Iterable
            all stock codes
        """

        raise NotImplementedError(f"Please implement the `get_all_stock` method")

    def get_data(
        self,
        stock_id: str,
        start_time: Union[pd.Timestamp, str],
        end_time: Union[pd.Timestamp, str],
        field: Union[str],
        method: Optional[str] = None,
    ) -> Union[None, int, float, bool, IndexData]:
        """获取指定股票在时间范围内的特定字段数据，并对数据应用指定方法

           示例:
            .. code-block::
                                        $close      $volume
                instrument  datetime
                SH600000    2010-01-04  86.778313   16162960.0
                            2010-01-05  87.433578   28117442.0
                            2010-01-06  85.713585   23632884.0
                            2010-01-07  83.788803   20813402.0
                            2010-01-08  84.730675   16044853.0

                SH600655    2010-01-04  2699.567383  158193.328125
                            2010-01-08  2612.359619   77501.406250
                            2010-01-11  2712.982422  160852.390625
                            2010-01-12  2788.688232  164587.937500
                            2010-01-13  2790.604004  145460.453125

                该方法有三种使用场景:

                1. method不为None时，返回int/float/bool/None
                    - 仅当方法返回None时，该方法才会返回None

                    示例:
                        print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method="last"))
                    输出: 85.713585

                2. method为None时，返回IndexData
                    示例:
                        print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method=None))
                    输出: IndexData([86.778313, 87.433578, 85.713585], [2010-01-04, 2010-01-05, 2010-01-06])

        参数
        ----------
        stock_id: str
            股票代码
        start_time : Union[pd.Timestamp, str]
            回测的闭区间开始时间
        end_time : Union[pd.Timestamp, str]
            回测的闭区间结束时间
        field : str
            要获取的数据列名
        method : Union[str, None]
            应用于数据的方法
            可选值: [None, "last", "all", "sum", "mean", "ts_data_last"]

        返回值
        ----------
        Union[None, int, float, bool, IndexData]
            在以下情况下会返回None:
            - 数据源中没有符合查询条件的股票数据
            - `method`方法返回None
        """

        raise NotImplementedError(f"Please implement the `get_data` method")


class PandasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument", group_keys=False):
            quote_dict[stock_id] = stock_val.droplevel(level="instrument")
        self.data = quote_dict

    def get_all_stock(self):
        return self.data.keys()

    def get_data(self, stock_id, start_time, end_time, field, method=None):
        if method == "ts_data_last":
            method = ts_data_last
        stock_data = resam_ts_data(self.data[stock_id][field], start_time, end_time, method=method)
        if stock_data is None:
            return None
        elif isinstance(stock_data, (bool, np.bool_, int, float, np.number)):
            return stock_data
        elif isinstance(stock_data, pd.Series):
            return idd.SingleData(stock_data)
        else:
            raise ValueError(f"stock data from resam_ts_data must be a number, pd.Series or pd.DataFrame")


class NumpyQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame, freq: str, region: str = "cn") -> None:
        """NumpyQuote初始化方法

        参数
        ----------
        quote_df : pd.DataFrame
            来自qlib的初始化数据框
        freq : str
            数据频率
        region : str
            地区代码，默认为'cn'
        
        属性
        ----------
        self.data : Dict(stock_id, IndexData.DataFrame)
            存储股票数据的字典，键为股票代码，值为IndexData.DataFrame
        """
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument", group_keys=False):
            quote_dict[stock_id] = idd.MultiData(stock_val.droplevel(level="instrument"))
            quote_dict[stock_id].sort_index()  # To support more flexible slicing, we must sort data first
        self.data = quote_dict

        n, unit = Freq.parse(freq)
        if unit in Freq.SUPPORT_CAL_LIST:
            self.freq = Freq.get_timedelta(1, unit)
        else:
            raise ValueError(f"{freq} is not supported in NumpyQuote")
        self.region = region

    def get_all_stock(self):
        return self.data.keys()

    @lru_cache(maxsize=512)
    def get_data(self, stock_id, start_time, end_time, field, method=None):
        # check stock id
        if stock_id not in self.get_all_stock():
            return None

        # 单个数据
        # 如果不考虑单个数据的分类，将会消耗大量时间
        if is_single_value(start_time, end_time, self.freq, self.region):
            # 这是一个非常特殊的情况
            # 跳过聚合函数以加快查询计算速度

            # 待修复：
            # 在以下情况下会进入 else 逻辑：
            # 1) 日线交易时的假期前一天
            # 2) 日内交易时的每天最后一分钟
            try:
                return self.data[stock_id].loc[start_time, field]
            except KeyError:
                return None
        else:
            data = self.data[stock_id].loc[start_time:end_time, field]
            if data.empty:
                return None
            if method is not None:
                data = self._agg_data(data, method)
            return data

    @staticmethod
    def _agg_data(data: IndexData, method: str) -> Union[IndexData, np.ndarray, None]:
        """根据特定方法聚合数据

        参数
        ----------
        data : IndexData
            待聚合的数据
        method : str
            聚合方法
            
        返回值
        ----------
        Union[IndexData, np.ndarray, None]
            聚合后的数据
        
        注意事项
        ----------
        - 为什么不直接调用数据的方法？
        """
        # FIXME: why not call the method of data directly?
        if method == "sum":
            return np.nansum(data)
        elif method == "mean":
            return np.nanmean(data)
        elif method == "last":
            # 待修复：我从未见过这个方法被调用
            # 请将其与 "ts_data_last" 合并
            return data[-1]
        elif method == "all":
            return data.all()
        elif method == "ts_data_last":
            valid_data = data.loc[~data.isna().data.astype(bool)]
            if len(valid_data) == 0:
                return None
            else:
                return valid_data.iloc[-1]
        else:
            raise ValueError(f"{method} is not supported")


class BaseSingleMetric:
    """
    The data structure of the single metric.
    The following methods are used for computing metrics in one indicator.
    """

    def __init__(self, metric: Union[dict, pd.Series]):
        """每个指标的单个数据结构。

        参数
        ----------
        metric : Union[dict, pd.Series]
            键/索引是股票代码，值是指标值。
            例如：
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
        """
        raise NotImplementedError(f"Please implement the `__init__` method")

    def __add__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__add__` method")

    def __radd__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        return self + other

    def __sub__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__sub__` method")

    def __rsub__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__rsub__` method")

    def __mul__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__mul__` method")

    def __truediv__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__truediv__` method")

    def __eq__(self, other: object) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__eq__` method")

    def __gt__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__gt__` method")

    def __lt__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `__lt__` method")

    def __len__(self) -> int:
        raise NotImplementedError(f"Please implement the `__len__` method")

    def sum(self) -> float:
        raise NotImplementedError(f"Please implement the `sum` method")

    def mean(self) -> float:
        raise NotImplementedError(f"Please implement the `mean` method")

    def count(self) -> int:
        """返回单个指标的计数，不包括 NaN 值。"""

        raise NotImplementedError(f"Please implement the `count` method")

    def abs(self) -> BaseSingleMetric:
        raise NotImplementedError(f"Please implement the `abs` method")

    @property
    def empty(self) -> bool:
        """如果指标为空，返回 True。"""

        raise NotImplementedError(f"Please implement the `empty` method")

    def add(self, other: BaseSingleMetric, fill_value: float = None) -> BaseSingleMetric:
        """用 fill_value 替换两个指标中的 np.nan 并将它们相加。"""

        raise NotImplementedError(f"Please implement the `add` method")

    def replace(self, replace_dict: dict) -> BaseSingleMetric:
        """根据 replace_dict 替换指标的值。"""

        raise NotImplementedError(f"Please implement the `replace` method")

    def apply(self, func: Callable) -> BaseSingleMetric:
        """用 func(metric) 替换指标的值。
        目前，func 仅限于 qlib/backtest/order/Order.parse_dir。
        """

        raise NotImplementedError(f"Please implement the 'apply' method")


class BaseOrderIndicator:
    """
    The data structure of order indicator.
    !!!NOTE: There are two ways to organize the data structure. Please choose a better way.
        1. One way is using BaseSingleMetric to represent each metric. For example, the data
        structure of PandasOrderIndicator is Dict[str, PandasSingleMetric]. It uses
        PandasSingleMetric based on pd.Series to represent each metric.
        2. The another way doesn't use BaseSingleMetric to represent each metric. The data
        structure of PandasOrderIndicator is a whole matrix. It means you are not necessary
        to inherit the BaseSingleMetric.
    """

    def __init__(self):
        self.data = {}  # will be created in the subclass
        self.logger = get_module_logger("online operator")

    def assign(self, col: str, metric: Union[dict, pd.Series]) -> None:
        """分配一个指标。

        参数
        ----------
        col : str
            指标的名称。
        metric : Union[dict, pd.Series]
            带有股票代码索引的指标，如成交金额、ffr等。
            例如：
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
        """

        raise NotImplementedError(f"Please implement the 'assign' method")

    def transfer(self, func: Callable, new_col: str = None) -> Optional[BaseSingleMetric]:
        """使用现有指标计算新指标。

        参数
        ----------
        func : Callable
            计算新指标的函数。
            函数的 kwargs 将在此函数中按名称替换为指标数据。
            例如：
                def func(pa):
                    return (pa > 0).sum() / pa.count()
        new_col : str, optional
            如果 new_col 不为 None，新指标将被分配到数据中，默认为 None。

        返回值
        ----------
        BaseSingleMetric
            新指标。
        """
        func_sig = inspect.signature(func).parameters.keys()
        func_kwargs = {sig: self.data[sig] for sig in func_sig}
        tmp_metric = func(**func_kwargs)
        if new_col is not None:
            self.data[new_col] = tmp_metric
            return None
        else:
            return tmp_metric

    def get_metric_series(self, metric: str) -> pd.Series:
        """以 pd.Series 格式返回单个指标。

        参数
        ----------
        metric : str
            指标名称。

        返回值
        ----------
        pd.Series
            单个指标。
            如果数据中没有该指标名称，返回 pd.Series()。
        """

        raise NotImplementedError(f"Please implement the 'get_metric_series' method")

    def get_index_data(self, metric: str) -> SingleData:
        """获取 SingleData 格式的单个指标

        参数
        ----------
        metric : str
            指标名称。

        返回值
        ------
        IndexData.Series
            SingleData 格式的单个指标
        """

        raise NotImplementedError(f"Please implement the 'get_index_data' method")

    @staticmethod
    def sum_all_indicators(
        order_indicator: BaseOrderIndicator,
        indicators: List[BaseOrderIndicator],
        metrics: Union[str, List[str]],
        fill_value: float = 0,
    ) -> None:
        """对具有相同指标的指标进行求和。
        并分配给订单指标(BaseOrderIndicator)。
        注意：当下层所有订单都失败时，indicators 可能是一个空列表。

        参数
        ----------
        order_indicator : BaseOrderIndicator
            要分配的订单指标。
        indicators : List[BaseOrderIndicator]
            所有内部指标的列表。
        metrics : Union[str, List[str]]
            需要求和的所有指标。
        fill_value : float, optional
            用该值填充 np.nan。默认为 None。
        """

        raise NotImplementedError(f"Please implement the 'sum_all_indicators' method")

    def to_series(self) -> Dict[Text, pd.Series]:
        """将指标以 pandas series 格式返回

        例如：{ "ffr":
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
                ...
         }
        """
        raise NotImplementedError(f"Please implement the `to_series` method")


class SingleMetric(BaseSingleMetric):
    def __init__(self, metric):
        self.metric = metric

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric + other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric + other.metric)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric - other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric - other.metric)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(other - self.metric)
        elif isinstance(other, self.__class__):
            return self.__class__(other.metric - self.metric)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric * other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric * other.metric)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric / other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric / other.metric)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric == other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric == other.metric)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric > other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric > other.metric)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric < other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric < other.metric)
        else:
            return NotImplemented

    def __len__(self):
        return len(self.metric)


class PandasSingleMetric(SingleMetric):
    """Each SingleMetric is based on pd.Series."""

    def __init__(self, metric: Union[dict, pd.Series] = {}):
        if isinstance(metric, dict):
            self.metric = pd.Series(metric)
        elif isinstance(metric, pd.Series):
            self.metric = metric
        else:
            raise ValueError(f"metric must be dict or pd.Series")

    def sum(self):
        return self.metric.sum()

    def mean(self):
        return self.metric.mean()

    def count(self):
        return self.metric.count()

    def abs(self):
        return self.__class__(self.metric.abs())

    @property
    def empty(self):
        return self.metric.empty

    @property
    def index(self):
        return list(self.metric.index)

    def add(self, other: BaseSingleMetric, fill_value: float = None) -> PandasSingleMetric:
        other = cast(PandasSingleMetric, other)
        return self.__class__(self.metric.add(other.metric, fill_value=fill_value))

    def replace(self, replace_dict: dict) -> PandasSingleMetric:
        return self.__class__(self.metric.replace(replace_dict))

    def apply(self, func: Callable) -> PandasSingleMetric:
        return self.__class__(self.metric.apply(func))

    def reindex(self, index: Any, fill_value: float) -> PandasSingleMetric:
        return self.__class__(self.metric.reindex(index, fill_value=fill_value))

    def __repr__(self):
        return repr(self.metric)


class PandasOrderIndicator(BaseOrderIndicator):
    """
    数据结构为OrderedDict(str: PandasSingleMetric)。
    每个基于pd.Series的PandasSingleMetric是一个指标。
    Str是指标名称。
    """

    def __init__(self) -> None:
        super(PandasOrderIndicator, self).__init__()
        self.data: Dict[str, PandasSingleMetric] = OrderedDict()

    def assign(self, col: str, metric: Union[dict, pd.Series]) -> None:
        self.data[col] = PandasSingleMetric(metric)

    def get_index_data(self, metric: str) -> SingleData:
        if metric in self.data:
            return idd.SingleData(self.data[metric].metric)
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if metric in self.data:
            return self.data[metric].metric
        else:
            return pd.Series()

    def to_series(self):
        return {k: v.metric for k, v in self.data.items()}

    @staticmethod
    def sum_all_indicators(
        order_indicator: BaseOrderIndicator,
        indicators: List[BaseOrderIndicator],
        metrics: Union[str, List[str]],
        fill_value: float = 0,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            tmp_metric = PandasSingleMetric({})
            for indicator in indicators:
                tmp_metric = tmp_metric.add(indicator.data[metric], fill_value)
            order_indicator.assign(metric, tmp_metric.metric)

    def __repr__(self):
        return repr(self.data)


class NumpyOrderIndicator(BaseOrderIndicator):
    """
    数据结构为OrderedDict(str: SingleData)。
    每个idd.SingleData是一个指标。
    Str是指标名称。
    """

    def __init__(self) -> None:
        super(NumpyOrderIndicator, self).__init__()
        self.data: Dict[str, SingleData] = OrderedDict()

    def assign(self, col: str, metric: dict) -> None:
        self.data[col] = idd.SingleData(metric)

    def get_index_data(self, metric: str) -> SingleData:
        if metric in self.data:
            return self.data[metric]
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        return self.data[metric].to_series()

    def to_series(self) -> Dict[str, pd.Series]:
        tmp_metric_dict = {}
        for metric in self.data:
            tmp_metric_dict[metric] = self.get_metric_series(metric)
        return tmp_metric_dict

    @staticmethod
    def sum_all_indicators(
        order_indicator: BaseOrderIndicator,
        indicators: List[BaseOrderIndicator],
        metrics: Union[str, List[str]],
        fill_value: float = 0,
    ) -> None:
        # 获取所有索引(stock_id)
        stock_set: set = set()
        for indicator in indicators:
            # set(np.ndarray.tolist()) is faster than set(np.ndarray)
            stock_set = stock_set | set(indicator.data[metrics[0]].index.tolist())
        stocks = sorted(list(stock_set))

        # 按索引添加指标
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            order_indicator.data[metric] = idd.sum_by_index(
                [indicator.data[metric] for indicator in indicators],
                stocks,
                fill_value,
            )

    def __repr__(self):
        return repr(self.data)
