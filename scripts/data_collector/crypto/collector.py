import abc
import sys
import datetime
from abc import ABC
from pathlib import Path

import fire
import pandas as pd
from loguru import logger
from dateutil.tz import tzlocal

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import deco_retry

from pycoingecko import CoinGeckoAPI
from time import mktime
from datetime import datetime as dt
import time


_CG_CRYPTO_SYMBOLS = None


def get_cg_crypto_symbols(qlib_data_path: [str, Path] = None) -> list:
    """获取CoinGecko中的加密货币代码

    返回值
    -------
        给定CoinGecko交易所列表中的加密货币代码
    """
    global _CG_CRYPTO_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_coingecko():
        try:
            cg = CoinGeckoAPI()
            resp = pd.DataFrame(cg.get_coins_markets(vs_currency="usd"))
        except Exception as e:
            raise ValueError("request error") from e
        try:
            _symbols = resp["id"].to_list()
        except Exception as e:
            logger.warning(f"request error: {e}")
            raise
        return _symbols

    if _CG_CRYPTO_SYMBOLS is None:
        _all_symbols = _get_coingecko()

        _CG_CRYPTO_SYMBOLS = sorted(set(_all_symbols))

    return _CG_CRYPTO_SYMBOLS


class CryptoCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=1,  # delay need to be one
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

    参数说明
    ----------
    save_dir: str
        加密货币数据保存目录
    max_workers: int
        工作线程数量，默认4
    max_collector_count: int
        默认2
    delay: float
        延迟时间（秒），默认0
    interval: str
        时间频率，取值为[1min, 1d]，默认1min
    start: str
        开始时间，默认None
    end: str
        结束时间，默认None
    check_data_length: int
        检查数据长度，如果不为None且大于0，当每个标的的数据长度大于等于该值时视为完整，否则将重新获取，最大获取次数为(max_collector_count)。默认None。
    limit_nums: int
        用于调试，默认None
    """
        super(CryptoCollector, self).__init__(
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
        raise NotImplementedError("重写获取时区方法")

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end):
        error_msg = f"{symbol}-{interval}-{start}-{end}"
        try:
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency="usd", days="max")
            _resp = pd.DataFrame(columns=["date"] + list(data.keys()))
            _resp["date"] = [dt.fromtimestamp(mktime(time.localtime(x[0] / 1000))) for x in data["prices"]]
            for key in data.keys():
                _resp[key] = [x[1] for x in data[key]]
            _resp["date"] = pd.to_datetime(_resp["date"])
            _resp["date"] = [x.date() for x in _resp["date"]]
            _resp = _resp[(_resp["date"] < pd.to_datetime(end).date()) & (_resp["date"] > pd.to_datetime(start).date())]
            if _resp.shape[0] != 0:
                _resp = _resp.reset_index()
            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
        except Exception as e:
            logger.warning(f"{error_msg}:{e}")

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> [pd.DataFrame]:
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = interval
            return self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )

        if interval == self.INTERVAL_1d:
            _result = _get_simple(start_datetime, end_datetime)
        else:
            raise ValueError(f"cannot support {interval}")
        return _result


class CryptoCollector1d(CryptoCollector, ABC):
    def get_instrument_list(self):
        logger.info("get coingecko crypto symbols......")
        symbols = get_cg_crypto_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class CryptoNormalize(BaseNormalize):
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def normalize_crypto(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        if df.empty:
            return df
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
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

        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalize_crypto(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        return df


class CryptoNormalize1d(CryptoNormalize):
    def _get_calendar_list(self):
        return None


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        """

        参数说明
        ----------
        source_dir: str
            从互联网收集的原始数据保存目录，默认"Path(__file__).parent/source"
        normalize_dir: str
            归一化数据目录，默认"Path(__file__).parent/normalize"
        max_workers: int
            并发数量，默认1
        interval: str
            时间频率，取值为[1min, 1d]，默认1d
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"CryptoCollector{self.interval}"

    @property
    def normalize_class_name(self):
        return f"CryptoNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
    ):
        """从互联网下载数据

        参数说明
        ----------
        max_collector_count: int
            默认2
        delay: float
            延迟时间（秒），默认0
        interval: str
            时间频率，取值为[1min, 1d]，默认1d，目前仅支持1d
        start: str
            开始时间，默认"2000-01-01"
        end: str
            结束时间，默认``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: int # 此参数是否有用？
            检查数据长度，如果不为None且大于0，当每个标的的数据长度大于等于该值时视为完整，否则将重新获取，最大获取次数为(max_collector_count)。默认None。
        limit_nums: int
            用于调试，默认None

        使用示例
        ---------
            # 获取日线数据
            $ python collector.py download_data --source_dir ~/.qlib/crypto_data/source/1d --start 2015-01-01 --end 2021-11-30 --delay 1 --interval 1d
        """

        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        """归一化数据

        参数说明
        ----------
        date_field_name: str
            日期字段名称，默认date
        symbol_field_name: str
            标的代码字段名称，默认symbol

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/crypto_data/source/1d --normalize_dir ~/.qlib/crypto_data/source/1d_nor --interval 1d --date_field_name date
        """
        super(Run, self).normalize_data(date_field_name, symbol_field_name)


if __name__ == "__main__":
    fire.Fire(Run)
