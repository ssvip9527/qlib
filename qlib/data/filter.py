# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import print_function
from abc import abstractmethod

import re
import pandas as pd
import numpy as np
import abc

from .data import Cal, DatasetD


class BaseDFilter(abc.ABC):
    """动态工具过滤器抽象类

    用户可以重写此类来构建自己的过滤器

    重写__init__方法以输入过滤规则

    重写filter_main方法以使用规则过滤工具
    """

    def __init__(self):
        pass

    @staticmethod
    def from_config(config):
        """从配置字典构造实例。

        参数
        ----------
        config : dict
            配置参数字典。
        """
        raise NotImplementedError("BaseDFilter的子类必须重写`from_config`方法")

    @abstractmethod
    def to_config(self):
        """将实例转换为配置字典。

        返回
        ----------
        dict
            返回配置参数字典。
        """
        raise NotImplementedError("BaseDFilter的子类必须重写`to_config`方法")


class SeriesDFilter(BaseDFilter):
    """动态工具过滤器抽象类，用于过滤特定特征的序列

    过滤器应提供以下参数：

    - 过滤开始时间
    - 过滤结束时间
    - 过滤规则

    重写__init__方法以分配特定规则来过滤序列。

    重写_getFilterSeries方法以使用规则过滤序列并获取{工具 => 序列}的字典，或重写filter_main以实现更高级的序列过滤规则
    """

    def __init__(self, fstart_time=None, fend_time=None, keep=False):
        """过滤器基类的初始化函数。
            在fstart_time和fend_time指定的时间段内，根据特定规则过滤一组工具。

        参数
        ----------
        fstart_time: str
            过滤规则开始过滤工具的时间。
        fend_time: str
            过滤规则停止过滤工具的时间。
        keep: bool
            是否保留在过滤时间范围内没有特征数据的工具。
        """
        super(SeriesDFilter, self).__init__()
        self.filter_start_time = pd.Timestamp(fstart_time) if fstart_time else None
        self.filter_end_time = pd.Timestamp(fend_time) if fend_time else None
        self.keep = keep

    def _getTimeBound(self, instruments):
        """获取所有工具的时间边界。

        参数
        ----------
        instruments: dict
            工具字典，格式为{工具名称 => 时间戳元组列表}。

        返回
        ----------
        pd.Timestamp, pd.Timestamp
            all instruments的下限时间和上限时间。
        """
        trange = Cal.calendar(freq=self.filter_freq)
        ubound, lbound = trange[0], trange[-1]
        for _, timestamp in instruments.items():
            if timestamp:
                lbound = timestamp[0][0] if timestamp[0][0] < lbound else lbound
                ubound = timestamp[-1][-1] if timestamp[-1][-1] > ubound else ubound
        return lbound, ubound

    def _toSeries(self, time_range, target_timestamp):
        """将目标时间戳转换为时区内的pandas布尔值序列。
            将target_timestamp范围内的时间设为TRUE，其他设为FALSE。

        参数
        ----------
        time_range : D.calendar
            工具的时间范围。
        target_timestamp : list
            时间戳元组列表(timestamp, timestamp)。

        返回
        ----------
        pd.Series
            工具的布尔值序列。
        """
        # Construct a whole dict of {date => bool}
        timestamp_series = {timestamp: False for timestamp in time_range}
        # Convert to pd.Series
        timestamp_series = pd.Series(timestamp_series)
        # Fill the date within target_timestamp with TRUE
        for start, end in target_timestamp:
            timestamp_series[Cal.calendar(start_time=start, end_time=end, freq=self.filter_freq)] = True
        return timestamp_series

    def _filterSeries(self, timestamp_series, filter_series):
        """通过两个序列的元素级AND运算，使用过滤序列过滤时间戳序列。

        参数
        ----------
        timestamp_series : pd.Series
            指示存在时间的布尔值序列。
        filter_series : pd.Series
            指示过滤特征的布尔值序列。

        返回
        ----------
        pd.Series
            指示日期是否满足过滤条件并存在于目标时间戳中的布尔值序列。
        """
        fstart, fend = list(filter_series.keys())[0], list(filter_series.keys())[-1]
        filter_series = filter_series.astype("bool")  # Make sure the filter_series is boolean
        timestamp_series[fstart:fend] = timestamp_series[fstart:fend] & filter_series
        return timestamp_series

    def _toTimestamp(self, timestamp_series):
        """将时间戳序列转换为指示连续TRUE范围的元组列表(timestamp, timestamp)。

        参数
        ----------
        timestamp_series: pd.Series
            过滤后的布尔值序列。

        返回
        ----------
        list
            时间戳元组列表(timestamp, timestamp)。
        """
        # sort the timestamp_series according to the timestamps
        timestamp_series.sort_index()
        timestamp = []
        _lbool = None
        _ltime = None
        _cur_start = None
        for _ts, _bool in timestamp_series.items():
            # there is likely to be NAN when the filter series don't have the
            # bool value, so we just change the NAN into False
            if _bool == np.nan:
                _bool = False
            if _lbool is None:
                _cur_start = _ts
                _lbool = _bool
                _ltime = _ts
                continue
            if (_lbool, _bool) == (True, False):
                if _cur_start:
                    timestamp.append((_cur_start, _ltime))
            elif (_lbool, _bool) == (False, True):
                _cur_start = _ts
            _lbool = _bool
            _ltime = _ts
        if _lbool:
            timestamp.append((_cur_start, _ltime))
        return timestamp

    def __call__(self, instruments, start_time=None, end_time=None, freq="day"):
        """调用此过滤器以获取过滤后的工具列表"""
        self.filter_freq = freq
        return self.filter_main(instruments, start_time, end_time)

    @abstractmethod
    def _getFilterSeries(self, instruments, fstart, fend):
        """根据初始化时分配的规则和输入时间范围获取过滤序列。

        参数
        ----------
        instruments : dict
            the dict of instruments to be filtered.
        fstart : pd.Timestamp
            start time of filter.
        fend : pd.Timestamp
            end time of filter.

        .. note:: fstart/fend indicates the intersection of instruments start/end time and filter start/end time.

        Returns
        ----------
        pd.Dataframe
            a series of {pd.Timestamp => bool}.
        """
        raise NotImplementedError("Subclass of SeriesDFilter must reimplement `getFilterSeries` method")

    def filter_main(self, instruments, start_time=None, end_time=None):
        """Implement this method to filter the instruments.

        Parameters
        ----------
        instruments: dict
            input instruments to be filtered.
        start_time: str
            start of the time range.
        end_time: str
            end of the time range.

        Returns
        ----------
        dict
            filtered instruments, same structure as input instruments.
        """
        lbound, ubound = self._getTimeBound(instruments)
        start_time = pd.Timestamp(start_time or lbound)
        end_time = pd.Timestamp(end_time or ubound)
        _instruments_filtered = {}
        _all_calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq=self.filter_freq)
        _filter_calendar = Cal.calendar(
            start_time=self.filter_start_time and max(self.filter_start_time, _all_calendar[0]) or _all_calendar[0],
            end_time=self.filter_end_time and min(self.filter_end_time, _all_calendar[-1]) or _all_calendar[-1],
            freq=self.filter_freq,
        )
        _all_filter_series = self._getFilterSeries(instruments, _filter_calendar[0], _filter_calendar[-1])
        for inst, timestamp in instruments.items():
            # Construct a whole map of date
            _timestamp_series = self._toSeries(_all_calendar, timestamp)
            # Get filter series
            if inst in _all_filter_series:
                _filter_series = _all_filter_series[inst]
            else:
                if self.keep:
                    _filter_series = pd.Series({timestamp: True for timestamp in _filter_calendar})
                else:
                    _filter_series = pd.Series({timestamp: False for timestamp in _filter_calendar})
            # Calculate bool value within the range of filter
            _timestamp_series = self._filterSeries(_timestamp_series, _filter_series)
            # Reform the map to (start_timestamp, end_timestamp) format
            _timestamp = self._toTimestamp(_timestamp_series)
            # Remove empty timestamp
            if _timestamp:
                _instruments_filtered[inst] = _timestamp
        return _instruments_filtered


class NameDFilter(SeriesDFilter):
    """Name dynamic instrument filter

    Filter the instruments based on a regulated name format.

    A name rule regular expression is required.
    """

    def __init__(self, name_rule_re, fstart_time=None, fend_time=None):
        """Init function for name filter class

        Parameters
        ----------
        name_rule_re: str
            regular expression for the name rule.
        """
        super(NameDFilter, self).__init__(fstart_time, fend_time)
        self.name_rule_re = name_rule_re

    def _getFilterSeries(self, instruments, fstart, fend):
        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        for inst, timestamp in instruments.items():
            if re.match(self.name_rule_re, inst):
                _filter_series = pd.Series({timestamp: True for timestamp in filter_calendar})
            else:
                _filter_series = pd.Series({timestamp: False for timestamp in filter_calendar})
            all_filter_series[inst] = _filter_series
        return all_filter_series

    @staticmethod
    def from_config(config):
        return NameDFilter(
            name_rule_re=config["name_rule_re"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
        )

    def to_config(self):
        return {
            "filter_type": "NameDFilter",
            "name_rule_re": self.name_rule_re,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
        }


class ExpressionDFilter(SeriesDFilter):
    """Expression dynamic instrument filter

    Filter the instruments based on a certain expression.

    An expression rule indicating a certain feature field is required.

    Examples
    ----------
    - *basic features filter* : rule_expression = '$close/$open>5'
    - *cross-sectional features filter* : rule_expression = '$rank($close)<10'
    - *time-sequence features filter* : rule_expression = '$Ref($close, 3)>100'
    """

    def __init__(self, rule_expression, fstart_time=None, fend_time=None, keep=False):
        """Init function for expression filter class

        Parameters
        ----------
        fstart_time: str
            filter the feature starting from this time.
        fend_time: str
            filter the feature ending by this time.
        rule_expression: str
            an input expression for the rule.
        """
        super(ExpressionDFilter, self).__init__(fstart_time, fend_time, keep=keep)
        self.rule_expression = rule_expression

    def _getFilterSeries(self, instruments, fstart, fend):
        # do not use dataset cache
        try:
            _features = DatasetD.dataset(
                instruments,
                [self.rule_expression],
                fstart,
                fend,
                freq=self.filter_freq,
                disk_cache=0,
            )
        except TypeError:
            # use LocalDatasetProvider
            _features = DatasetD.dataset(instruments, [self.rule_expression], fstart, fend, freq=self.filter_freq)
        rule_expression_field_name = list(_features.keys())[0]
        all_filter_series = _features[rule_expression_field_name]
        return all_filter_series

    @staticmethod
    def from_config(config):
        return ExpressionDFilter(
            rule_expression=config["rule_expression"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
            keep=config["keep"],
        )

    def to_config(self):
        return {
            "filter_type": "ExpressionDFilter",
            "rule_expression": self.rule_expression,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
            "keep": self.keep,
        }
