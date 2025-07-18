import numpy as np
import pandas as pd

from functools import partial
from typing import Union, Callable

from . import lazy_sort_index
from .time import Freq, cal_sam_minute
from ..config import C


def resam_calendar(
    calendar_raw: np.ndarray, freq_raw: Union[str, Freq], freq_sam: Union[str, Freq], region: str = None
) -> np.ndarray:
    """
    将频率为freq_raw的日历重采样为频率为freq_sam的日历
    假设条件:
        - 每日日历长度固定为240

    参数
    ----------
    calendar_raw : np.ndarray
        原始频率为freq_raw的日历
    freq_raw : str
        原始日历频率
    freq_sam : str
        采样频率
    region: str
        地区，例如"cn"(中国),"us"(美国)

    返回值
    -------
    np.ndarray
        频率为freq_sam的日历
    """
    if region is None:
        region = C["region"]

    freq_raw = Freq(freq_raw)
    freq_sam = Freq(freq_sam)
    if not len(calendar_raw):
        return calendar_raw

    # if freq_sam is xminute, divide each trading day into several bars evenly
    if freq_sam.base == Freq.NORM_FREQ_MINUTE:
        if freq_raw.base != Freq.NORM_FREQ_MINUTE:
            raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
        else:
            if freq_raw.count > freq_sam.count:
                raise ValueError("raw freq must be higher than sampling freq")
        _calendar_minute = np.unique(list(map(lambda x: cal_sam_minute(x, freq_sam.count, region), calendar_raw)))
        return _calendar_minute

    # else, convert the raw calendar into day calendar, and divide the whole calendar into several bars evenly
    else:
        _calendar_day = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, 0, 0, 0), calendar_raw)))
        if freq_sam.base == Freq.NORM_FREQ_DAY:
            return _calendar_day[:: freq_sam.count]

        elif freq_sam.base == Freq.NORM_FREQ_WEEK:
            _day_in_week = np.array(list(map(lambda x: x.dayofweek, _calendar_day)))
            _calendar_week = _calendar_day[np.ediff1d(_day_in_week, to_begin=-1) < 0]
            return _calendar_week[:: freq_sam.count]

        elif freq_sam.base == Freq.NORM_FREQ_MONTH:
            _day_in_month = np.array(list(map(lambda x: x.day, _calendar_day)))
            _calendar_month = _calendar_day[np.ediff1d(_day_in_month, to_begin=-1) < 0]
            return _calendar_month[:: freq_sam.count]
        else:
            raise ValueError("sampling freq must be xmin, xd, xw, xm")


def get_higher_eq_freq_feature(instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1):
    """获取频率高于或等于`freq`的特征

    返回值
    -------
    pd.DataFrame
        频率高于或等于指定频率的特征
    """

    from ..data.data import D  # pylint: disable=C0415

    try:
        _result = D.features(instruments, fields, start_time, end_time, freq=freq, disk_cache=disk_cache)
        _freq = freq
    except (ValueError, KeyError) as value_key_e:
        _, norm_freq = Freq.parse(freq)
        if norm_freq in [Freq.NORM_FREQ_MONTH, Freq.NORM_FREQ_WEEK, Freq.NORM_FREQ_DAY]:
            try:
                _result = D.features(instruments, fields, start_time, end_time, freq="day", disk_cache=disk_cache)
                _freq = "day"
            except (ValueError, KeyError):
                _result = D.features(instruments, fields, start_time, end_time, freq="1min", disk_cache=disk_cache)
                _freq = "1min"
        elif norm_freq == Freq.NORM_FREQ_MINUTE:
            _result = D.features(instruments, fields, start_time, end_time, freq="1min", disk_cache=disk_cache)
            _freq = "1min"
        else:
            raise ValueError(f"freq {freq} is not supported") from value_key_e
    return _result, _freq


def resam_ts_data(
    ts_feature: Union[pd.DataFrame, pd.Series],
    start_time: Union[str, pd.Timestamp] = None,
    end_time: Union[str, pd.Timestamp] = None,
    method: Union[str, Callable] = "last",
    method_kwargs: dict = {},
):
    """从时间序列数据中重采样值

        - 如果`feature`具有MultiIndex[instrument, datetime]，则对[start_time, end_time]区间内每个instrument的数据应用`method`
            示例:

            .. code-block::

                print(feature)
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

                print(resam_ts_data(feature, start_time="2010-01-04", end_time="2010-01-05", fields=["$close", "$volume"], method="last"))
                            $close      $volume
                instrument
                SH600000    87.433578 28117442.0
                SH600655    2699.567383  158193.328125

        - 否则，`feature`应具有Index[datetime]，直接对`feature`应用`method`
            示例:

            .. code-block::
                print(feature)
                            $close      $volume
                datetime
                2010-01-04  86.778313   16162960.0
                2010-01-05  87.433578   28117442.0
                2010-01-06  85.713585   23632884.0
                2010-01-07  83.788803   20813402.0
                2010-01-08  84.730675   16044853.0

                print(resam_ts_data(feature, start_time="2010-01-04", end_time="2010-01-05", method="last"))

                $close 87.433578
                $volume 28117442.0

                print(resam_ts_data(feature['$close'], start_time="2010-01-04", end_time="2010-01-05", method="last"))

                87.433578

    参数
    ----------
    ts_feature : Union[pd.DataFrame, pd.Series]
        待重采样的原始时间序列特征
    start_time : Union[str, pd.Timestamp], optional
        采样开始时间，默认为None
    end_time : Union[str, pd.Timestamp], optional
        采样结束时间，默认为None
    method : Union[str, Callable], optional
        采样方法，对每个股票序列数据应用方法函数，默认为"last"
        - 如果method是字符串或可调用函数，应为SeriesGroupBy或DataFrameGroupby的属性，并对切片后的时间序列数据应用groupy.method
        - 如果method为None，则不对切片后的时间序列数据做任何处理
    method_kwargs : dict, optional
        方法的参数，默认为{}

    返回值
    -------
        重采样后的DataFrame/Series/值，当重采样数据为空时返回None
    """

    selector_datetime = slice(start_time, end_time)

    from ..data.dataset.utils import get_level_index  # pylint: disable=C0415

    feature = lazy_sort_index(ts_feature)

    datetime_level = get_level_index(feature, level="datetime") == 0
    if datetime_level:
        feature = feature.loc[selector_datetime]
    else:
        feature = feature.loc(axis=0)[(slice(None), selector_datetime)]

    if feature.empty:
        return None
    if isinstance(feature.index, pd.MultiIndex):
        if callable(method):
            method_func = method
            return feature.groupby(level="instrument", group_keys=False).apply(method_func, **method_kwargs)
        elif isinstance(method, str):
            return getattr(feature.groupby(level="instrument", group_keys=False), method)(**method_kwargs)
    else:
        if callable(method):
            method_func = method
            return method_func(feature, **method_kwargs)
        elif isinstance(method, str):
            return getattr(feature, method)(**method_kwargs)
    return feature


def get_valid_value(series, last=True):
    """获取单级索引pd.Series的第一个/最后一个非NaN值

    参数
    ----------
    series : pd.Series
        序列不应为空
    last : bool, optional
        是否获取最后一个有效值，默认为True
        - 如果last为True，获取最后一个有效值
        - 否则，获取第一个有效值

    返回值
    -------
    Nan | float
        `first/last`有效值
    """
    return series.fillna(method="ffill").iloc[-1] if last else series.fillna(method="bfill").iloc[0]


def _ts_data_valid(ts_feature, last=False):
    """获取单级索引pd.Series|DataFrame的第一个/最后一个非NaN值"""
    if isinstance(ts_feature, pd.DataFrame):
        return ts_feature.apply(lambda column: get_valid_value(column, last=last))
    elif isinstance(ts_feature, pd.Series):
        return get_valid_value(ts_feature, last=last)
    else:
        raise TypeError(f"ts_feature should be pd.DataFrame/Series, not {type(ts_feature)}")


ts_data_last = partial(_ts_data_valid, last=True)
ts_data_first = partial(_ts_data_valid, last=False)
