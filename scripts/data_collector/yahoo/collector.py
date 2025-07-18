# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    get_in_stock_symbols,
    get_br_stock_symbols,
    generate_minutes_calendar_from_daily,
    calc_adjusted_price,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class YahooCollector(BaseCollector):
    retry = 5  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(YahooCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        def _show_logging_func():
            if interval == YahooCollector.INTERVAL_1min and show_1min_logging:
                logger.warning(f"{error_msg}:{_resp}")

        interval = "1m" if interval in ["1m", "1min"] else interval
        try:
            _resp = Ticker(symbol, asynchronous=False).history(interval=interval, start=start, end=end)
            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
            elif isinstance(_resp, dict):
                _temp_data = _resp.get(symbol, {})
                if isinstance(_temp_data, str) or (
                    isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
                ):
                    _show_logging_func()
            else:
                _show_logging_func()
        except Exception as e:
            logger.warning(
                f"get data error: {symbol}--{start}--{end}"
                + "Your data request fails. This may be caused by your firewall (e.g. GFW). Please switch your network if you want to access Yahoo! data"
            )

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            resp = self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}" + "The stock may be delisted, please check"
                )
            return resp

        _result = None
        if interval == self.INTERVAL_1d:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                pass
        elif interval == self.INTERVAL_1min:
            _res = []
            _start = self.start_datetime
            while _start < self.end_datetime:
                _tmp_end = min(_start + pd.Timedelta(days=7), self.end_datetime)
                try:
                    _resp = _get_simple(_start, _tmp_end)
                    _res.append(_resp)
                except ValueError as e:
                    pass
                _start = _tmp_end
            if _res:
                _result = pd.concat(_res, sort=False).sort_values(["symbol", "date"])
        else:
            raise ValueError(f"cannot support {self.interval}")
        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(YahooCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")


class YahooCollectorCN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class YahooCollectorCN1d(YahooCollectorCN):
    def download_index_data(self):
        # TODO: 来自MSN
        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903", "csi500": "000905"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(
                            INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end), timeout=None
                        ).json()["data"]["klines"],
                    )
                )
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}"
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)


class YahooCollectorCN1min(YahooCollectorCN):
    def get_instrument_list(self):
        symbols = super(YahooCollectorCN1min, self).get_instrument_list()
        return symbols + ["000300.ss", "000905.ss", "000903.ss"]

    def download_index_data(self):
        pass


class YahooCollectorUS(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get US stock symbols......")
        symbols = get_us_stock_symbols() + [
            "^GSPC",
            "^NDX",
            "^DJI",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "America/New_York"


class YahooCollectorUS1d(YahooCollectorUS):
    pass


class YahooCollectorUS1min(YahooCollectorUS):
    pass


class YahooCollectorIN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get INDIA stock symbols......")
        symbols = get_in_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "Asia/Kolkata"


class YahooCollectorIN1d(YahooCollectorIN):
    pass


class YahooCollectorIN1min(YahooCollectorIN):
    pass


class YahooCollectorBR(YahooCollector, ABC):
    def retry(cls):  # pylint: disable=E0213
        """
        使用retry=2的原因是，
        遗憾的是Yahoo Finance没有跟踪一些巴西股票。

        因此，将retry参数设置为5的deco_retry装饰器将尝试获取股票数据最多5次，
        这会使下载巴西股票的代码非常慢。

        将来这可能会改变，但目前
        建议将retry参数保留为1或2以提高下载速度。

        为实现此目标，在YahooCollectorBR基类中添加了抽象属性(retry)
        """
        raise NotImplementedError

    def get_instrument_list(self):
        logger.info("get BR stock symbols......")
        symbols = get_br_stock_symbols() + [
            "^BVSP",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "Brazil/East"


class YahooCollectorBR1d(YahooCollectorBR):
    retry = 2


class YahooCollectorBR1min(YahooCollectorBR):
    retry = 2


class YahooNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_yahoo(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(YahooNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        change_series = YahooNormalize.calc_change(df, last_close)
        # 注意：Yahoo Finance获取的数据有时会有异常
        # 警告：如果某个`标的(交易所)`连续交易日的价格差异达到89到111倍是正常情况，
        # 警告：则需要修改以下行的逻辑
        _count = 0
        while True:
            # 注意：可能会连续多天出现异常
            change_series = YahooNormalize.calc_change(df, last_close)
            _mask = (change_series >= 89) & (change_series <= 111)
            if not _mask.any():
                break
            _tmp_cols = ["high", "close", "low", "open", "adjclose"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100
            _count += 1
            if _count >= 10:
                _symbol = df.loc[df[symbol_field_name].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        df["change"] = YahooNormalize.calc_change(df, last_close)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_yahoo(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """adjusted price"""
        raise NotImplementedError("rewrite adjusted_price")


class YahooNormalize1d(YahooNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].fillna(method="ffill")
        else:
            df["factor"] = 1
        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """获取首个收盘价

        注意事项
        -----
            对于Yahoo日线数据的增量更新（追加），用户需要使用现有数据首个交易日非零的收盘价
        """
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        return _close

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """手动调整数据：所有字段（除change外）均按首个交易日收盘价进行标准化"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)
        for _col in df.columns:
            # NOTE: retain original adjclose, required for incremental updates
            if _col in [self._symbol_field_name, "adjclose", "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()


class YahooNormalize1dExtend(YahooNormalize1d):
    def __init__(
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """
        参数说明
        ----------
        old_qlib_data_dir: str, Path
            用于Yahoo数据更新的QLib历史数据目录，通常来自：https://github.com/ssvip9527/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            日期字段名称，默认为date
        symbol_field_name: str
            标的代码字段名称，默认为symbol
        """
        super(YahooNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1dExtend, self).normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index()


class YahooNormalize1min(YahooNormalize, ABC):
    """使用本地日线数据归一化到1分钟数据"""

    AM_RANGE = None  # type: tuple  # eg: ("09:30:00", "11:29:00")
    PM_RANGE = None  # type: tuple  # eg: ("13:00:00", "14:59:00")

    # Whether the trading day of 1min data is consistent with 1d
    CONSISTENT_1d = True
    CALC_PAUSED_NUM = True

    def __init__(
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        参数说明
        ----------
        qlib_data_1d_dir: str, Path
            用于Yahoo数据更新的QLib日线数据目录，通常用于：使用本地日线数据归一化到1分钟数据
        date_field_name: str
            日期字段名称，默认为date
        symbol_field_name: str
            标的代码字段名称，默认为symbol
        """
        super(YahooNormalize1min, self).__init__(date_field_name, symbol_field_name)
        qlib.init(provider_uri=qlib_data_1d_dir)
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return list(D.calendar(freq="day"))

    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    def generate_1min_from_daily(self, calendars: Iterable) -> pd.Index:
        """从日线日历生成1分钟级交易日历"""
        return generate_minutes_calendar_from_daily(
            calendars, freq="1min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算1分钟数据的复权价格"""
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="1min",
            consistent_1d=self.CONSISTENT_1d,
            calc_paused=self.CALC_PAUSED_NUM,
            _1d_data_all=self.all_1d_data,
        )
        return df

    @abc.abstractmethod
    def symbol_to_yahoo(self, symbol):
        """将标的代码转换为Yahoo Finance格式"""
        raise NotImplementedError("需重写symbol_to_yahoo方法")

    @abc.abstractmethod
    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        """获取日线级交易日历列表"""
        raise NotImplementedError("需重写_get_1d_calendar_list方法")


class YahooNormalizeUS:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("US_ALL")


class YahooNormalizeUS1d(YahooNormalizeUS, YahooNormalize1d):
    pass


class YahooNormalizeUS1dExtend(YahooNormalizeUS, YahooNormalize1dExtend):
    pass


class YahooNormalizeUS1min(YahooNormalizeUS, YahooNormalize1min):
    """美国市场1分钟数据归一化器"""
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: 支持1分钟数据
        raise ValueError("不支持1分钟数据")

    def _get_1d_calendar_list(self):
        """获取美国市场日线交易日历"""
        return get_calendar_list("US_ALL")

    def symbol_to_yahoo(self, symbol):
        """将美国市场标的代码转换为Yahoo格式"""
        return fname_to_code(symbol)


class YahooNormalizeIN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("IN_ALL")


class YahooNormalizeIN1d(YahooNormalizeIN, YahooNormalize1d):
    pass


class YahooNormalizeIN1min(YahooNormalizeIN, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("IN_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeCN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("ALL")


class YahooNormalizeCN1d(YahooNormalizeCN, YahooNormalize1d):
    pass


class YahooNormalizeCN1dExtend(YahooNormalizeCN, YahooNormalize1dExtend):
    pass


class YahooNormalizeCN1min(YahooNormalizeCN, YahooNormalize1min):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_1min_from_daily(self.calendar_list_1d)

    def symbol_to_yahoo(self, symbol):
        if "." not in symbol:
            _exchange = symbol[:2]
            _exchange = ("ss" if _exchange.islower() else "SS") if _exchange.lower() == "sh" else _exchange
            symbol = symbol[2:] + "." + _exchange
        return symbol

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")


class YahooNormalizeBR:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("BR_ALL")


class YahooNormalizeBR1d(YahooNormalizeBR, YahooNormalize1d):
    pass


class YahooNormalizeBR1min(YahooNormalizeBR, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("BR_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        """

        参数说明
        ----------
        source_dir: str
            从互联网收集的原始数据保存目录，默认为"Path(__file__).parent/source"
        normalize_dir: str
            归一化后数据保存目录，默认为"Path(__file__).parent/normalize"
        max_workers: int
            并发数量，默认为1；收集数据时建议将max_workers设置为1
        interval: str
            时间频率，取值为[1min, 1d]，默认为1d
        region: str
            市场区域，取值为["CN", "US", "BR"]，默认为"CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"YahooCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"YahooNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """从互联网下载数据

        参数说明
        ----------
        max_collector_count: int
            默认值为2
        delay: float
            延迟时间（秒），默认值为0.5
        start: str
            开始时间，默认为"2000-01-01"；闭区间（包含开始时间）
        end: str
            结束时间，默认为``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``；开区间（不包含结束时间）
        check_data_length: int
            检查数据长度，如果不为None且大于0，每个标的数据长度大于等于该值则认为完整，否则将重新获取，最大获取次数为(max_collector_count)。默认值为None
        limit_nums: int
            用于调试，默认值为None

        注意事项
        -----
            check_data_length参数示例：
                日线数据，一年约：252 // 4
                美国1分钟数据，一周约：6.5 * 60 * 5
                中国1分钟数据，一周约：4 * 60 * 5

        使用示例
        ---------
            # 获取日线数据
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # 获取1分钟数据
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        if self.interval == "1d" and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            raise ValueError(f"end_date: {end} is greater than the current date.")

        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """归一化数据

        参数说明
        ----------
        date_field_name: str
            日期字段名称，默认为date
        symbol_field_name: str
            标的代码字段名称，默认为symbol
        end_date: str
            若不为None，则归一化保存的最后日期（包含end_date）；若为None，则忽略此参数；默认值为None
        qlib_data_1d_dir: str
            当interval==1min时，qlib_data_1d_dir不能为空，归一化1分钟数据需要使用日线数据；

                qlib日线数据可通过以下方式获取：
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                或者：
                    下载日线数据，参考：https://github.com/ssvip9527/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        使用示例
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/ssvip9527/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """归一化数据扩展；扩展Yahoo Qlib数据(来源: https://github.com/ssvip9527/qlib/tree/main/scripts#download-cn-data)

        注意事项
        -----
            扩展Yahoo Qlib数据的步骤：

                1. 下载Qlib数据：https://github.com/ssvip9527/qlib/tree/main/scripts#download-cn-data；保存至<dir1>

                2. 收集源数据：https://github.com/ssvip9527/qlib/tree/main/scripts/data_collector/yahoo#collector-data；保存至<dir2>

                3. 归一化新源数据(来自步骤2)：python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

                4. 导出数据：python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

                5. 更新标的(例如：沪深300)：python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

        参数说明
        ----------
        old_qlib_data_dir: str
            用于更新的Yahoo Qlib数据，通常来自：https://github.com/ssvip9527/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            日期字段名称，默认为date
        symbol_field_name: str
            标的代码字段名称，默认为symbol

        使用示例
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """从互联网下载今日数据

        参数说明
        ----------
        max_collector_count: int
            默认为2
        delay: float
            时间间隔(秒)，time.sleep(delay)，默认为0.5
        check_data_length: int
            检查数据长度，如果不为None且大于0，当每个标的的数据长度大于等于该值时视为完整，否则将重新获取，最大获取次数为(max_collector_count)。默认为None。
        limit_nums: int
            用于调试，默认为None

        注意事项
        -----
            下载今日数据：
                start_time = datetime.datetime.now().date(); 闭区间(包含开始时间)
                end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); 开区间(不包含结束时间)

            check_data_length示例：
                日线数据，一年：252 // 4
                美国市场1分钟数据，一周：6.5 * 60 * 5
                中国市场1分钟数据，一周：4 * 60 * 5

        使用示例
        ---------
            # 获取日线数据
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
            # 获取1分钟数据
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = False,
    ):
        """将Yahoo数据更新为二进制格式

        参数说明
        ----------
        qlib_data_1d_dir: str
            用于Yahoo数据更新的Qlib数据，通常来自：https://github.com/ssvip9527/qlib/tree/main/scripts#download-cn-data

        end_date: str
            结束日期时间，默认为``pd.Timestamp(trading_date + pd.Timedelta(days=1))``；开区间(不包含结束日期)
        check_data_length: int
            检查数据长度，如果不为None且大于0，当每个标的的数据长度大于等于该值时视为完整，否则将重新获取，最大获取次数为(max_collector_count)。默认为None。
        delay: float
            时间间隔(秒)，time.sleep(delay)，默认为1
        exists_skip: bool
            是否跳过已存在文件，默认为False
        注意事项
        -----
            如果qlib_data_dir中的数据不完整，将使用np.nan填充前一交易日的trading_date

        使用示例
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <用户数据目录> --trading_date <开始日期> --end_date <结束日期>
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # download qlib 1d data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        # start/end date
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # download data from yahoo
        # NOTE: when downloading data from YahooFinance, max_workers is recommended to be 1
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)
