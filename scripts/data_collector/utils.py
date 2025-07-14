#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import copy
import importlib
import time
import bisect
import pickle
import random
import requests
import functools
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup

HS_SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"

CALENDAR_URL_BASE = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20991231"
SZSE_CALENDAR_URL = "http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={month}&random={random}"

CALENDAR_BENCH_URL_MAP = {
    "CSI300": CALENDAR_URL_BASE.format(market=1, bench_code="000300"),
    "CSI500": CALENDAR_URL_BASE.format(market=1, bench_code="000905"),
    "CSI100": CALENDAR_URL_BASE.format(market=1, bench_code="000903"),
    # NOTE: Use the time series of SH600000 as the sequence of all stocks
    "ALL": CALENDAR_URL_BASE.format(market=1, bench_code="000905"),
    # NOTE: Use the time series of ^GSPC(SP500) as the sequence of all stocks
    "US_ALL": "^GSPC",
    "IN_ALL": "^NSEI",
    "BR_ALL": "^BVSP",
}

_BENCH_CALENDAR_LIST = None
_ALL_CALENDAR_LIST = None
_HS_SYMBOLS = None
_US_SYMBOLS = None
_IN_SYMBOLS = None
_BR_SYMBOLS = None
_EN_FUND_SYMBOLS = None
_CALENDAR_MAP = {}

# NOTE: Until 2020-10-20 20:00:00
MINIMUM_SYMBOLS_NUM = 3900


def get_calendar_list(bench_code="CSI300") -> List[pd.Timestamp]:
    """获取沪深历史交易日历列表

    参数说明
    ----------
    bench_code: str
        有效值包括["CSI300", "CSI500", "ALL", "US_ALL"]

    返回值
    -------
        历史交易日历列表
    """

    logger.info(f"get calendar list: {bench_code}......")

    def _get_calendar(url):
        _value_list = requests.get(url, timeout=None).json()["data"]["klines"]
        return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))

    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        if bench_code.startswith("US_") or bench_code.startswith("IN_") or bench_code.startswith("BR_"):
            print(Ticker(CALENDAR_BENCH_URL_MAP[bench_code]))
            print(Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max"))
            df = Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max")
            calendar = df.index.get_level_values(level="date").map(pd.Timestamp).unique().tolist()
        else:
            if bench_code.upper() == "ALL":

                @deco_retry
                def _get_calendar(month):
                    _cal = []
                    try:
                        resp = requests.get(
                            SZSE_CALENDAR_URL.format(month=month, random=random.random), timeout=None
                        ).json()
                        for _r in resp["data"]:
                            if int(_r["jybz"]):
                                _cal.append(pd.Timestamp(_r["jyrq"]))
                    except Exception as e:
                        raise ValueError(f"{month}-->{e}") from e
                    return _cal

                month_range = pd.date_range(start="2000-01", end=pd.Timestamp.now() + pd.Timedelta(days=31), freq="M")
                calendar = []
                for _m in month_range:
                    cal = _get_calendar(_m.strftime("%Y-%m"))
                    if cal:
                        calendar += cal
                calendar = list(filter(lambda x: x <= pd.Timestamp.now(), calendar))
            else:
                calendar = _get_calendar(CALENDAR_BENCH_URL_MAP[bench_code])
        _CALENDAR_MAP[bench_code] = calendar
    logger.info(f"end of get calendar list: {bench_code}.")
    return calendar


def return_date_list(date_field_name: str, file_path: Path):
    """返回日期列表

    参数说明
    ----------
    date_field_name: str
        日期字段名称
    file_path: Path
        文件路径

    返回值
    -------
        排序后的日期列表
    """
    date_list = pd.read_csv(file_path, sep=",", index_col=0)[date_field_name].to_list()
    return sorted([pd.Timestamp(x) for x in date_list])


def get_calendar_list_by_ratio(
    source_dir: [str, Path],
    date_field_name: str = "date",
    threshold: float = 0.5,
    minimum_count: int = 10,
    max_workers: int = 16,
) -> list:
    """通过选择基金交易较少的日期获取交易日历列表

    参数说明
    ----------
    source_dir: str or Path
        从互联网收集的原始数据保存目录
    date_field_name: str
            日期字段名称，默认为'date'
    threshold: float
        排除基金交易较少日期的阈值，默认为0.5
    minimum_count: int
        单日基金交易的最小数量
    max_workers: int
        并发数量，默认为16

    返回值
    -------
        历史交易日历列表
    """
    logger.info(f"get calendar list from {source_dir} by threshold = {threshold}......")

    source_dir = Path(source_dir).expanduser()
    file_list = list(source_dir.glob("*.csv"))

    _number_all_funds = len(file_list)

    logger.info(f"count how many funds trade in this day......")
    _dict_count_trade = dict()  # dict{date:count}
    _fun = partial(return_date_list, date_field_name)
    all_oldest_list = []
    with tqdm(total=_number_all_funds) as p_bar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for date_list in executor.map(_fun, file_list):
                if date_list:
                    all_oldest_list.append(date_list[0])
                for date in date_list:
                    if date not in _dict_count_trade:
                        _dict_count_trade[date] = 0

                    _dict_count_trade[date] += 1

                p_bar.update()

    logger.info(f"count how many funds have founded in this day......")
    _dict_count_founding = {date: _number_all_funds for date in _dict_count_trade}  # dict{date:count}
    with tqdm(total=_number_all_funds) as p_bar:
        for oldest_date in all_oldest_list:
            for date in _dict_count_founding.keys():
                if date < oldest_date:
                    _dict_count_founding[date] -= 1

    calendar = [
        date for date, count in _dict_count_trade.items() if count >= max(int(count * threshold), minimum_count)
    ]

    return calendar


def get_hs_stock_symbols() -> list:
    """获取沪深股票代码

    返回值
    -------
        股票代码列表
    """
    global _HS_SYMBOLS  # pylint: disable=W0603

    def _get_symbol():
        """
        Get the stock pool from a web page and process it into the format required by yahooquery.
        Format of data retrieved from the web page: 600519, 000001
        The data format required by yahooquery: 600519.ss, 000001.sz

        Returns
        -------
            set: Returns the set of symbol codes.

        Examples:
        -------
            {600000.ss, 600001.ss, 600002.ss, 600003.ss, ...}
        """
        # url = "http://99.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12"

        base_url = "http://99.push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1,  # page number
            "pz": 100,  # page size, default to 100
            "po": 1,
            "np": 1,
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "fields": "f12",
        }

        _symbols = []
        page = 1

        while True:
            params["pn"] = page
            try:
                resp = requests.get(base_url, params=params, timeout=None)
                resp.raise_for_status()
                data = resp.json()

                # Check if response contains valid data
                if not data or "data" not in data or not data["data"] or "diff" not in data["data"]:
                    logger.warning(f"Invalid response structure on page {page}")
                    break

                # fetch the current page data
                current_symbols = [_v["f12"] for _v in data["data"]["diff"]]

                if not current_symbols:  # It's the last page if there is no data in current page
                    logger.info(f"Last page reached: {page - 1}")
                    break

                _symbols.extend(current_symbols)

                # show progress
                logger.info(
                    f"Page {page}: fetch {len(current_symbols)} stocks:[{current_symbols[0]} ... {current_symbols[-1]}]"
                )

                page += 1

                # sleep time to avoid overloading the server
                time.sleep(0.5)

            except requests.exceptions.HTTPError as e:
                raise requests.exceptions.HTTPError(
                    f"Request to {base_url} failed with status code {resp.status_code}"
                ) from e
            except Exception as e:
                logger.warning("An error occurred while extracting data from the response.")
                raise

        if len(_symbols) < 3900:
            raise ValueError("The complete list of stocks is not available.")

        # Add suffix after the stock code to conform to yahooquery standard, otherwise the data will not be fetched.
        _symbols = [
            _symbol + ".ss" if _symbol.startswith("6") else _symbol + ".sz" if _symbol.startswith(("0", "3")) else None
            for _symbol in _symbols
        ]
        _symbols = [_symbol for _symbol in _symbols if _symbol is not None]

        return set(_symbols)

    if _HS_SYMBOLS is None:
        symbols = set()
        _retry = 60
        # It may take multiple times to get the complete
        while len(symbols) < MINIMUM_SYMBOLS_NUM:
            symbols |= _get_symbol()
            time.sleep(3)

        symbol_cache_path = Path("~/.cache/hs_symbols_cache.pkl").expanduser().resolve()
        symbol_cache_path.parent.mkdir(parents=True, exist_ok=True)
        if symbol_cache_path.exists():
            with symbol_cache_path.open("rb") as fp:
                cache_symbols = pickle.load(fp)
                symbols |= cache_symbols
        with symbol_cache_path.open("wb") as fp:
            pickle.dump(symbols, fp)

        _HS_SYMBOLS = sorted(list(symbols))

    return _HS_SYMBOLS


def get_us_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """获取美国股票代码

    返回值
    -------
        股票代码列表
    """
    global _US_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_eastmoney():
        url = "http://4.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&fs=m:105,m:106,m:107&fields=f12"
        resp = requests.get(url, timeout=None)
        if resp.status_code != 200:
            raise ValueError("request error")

        try:
            _symbols = [_v["f12"].replace("_", "-P") for _v in resp.json()["data"]["diff"].values()]
        except Exception as e:
            logger.warning(f"request error: {e}")
            raise

        if len(_symbols) < 8000:
            raise ValueError("request error")

        return _symbols

    @deco_retry
    def _get_nasdaq():
        _res_symbols = []
        for _name in ["otherlisted", "nasdaqtraded"]:
            url = f"ftp://ftp.nasdaqtrader.com/SymbolDirectory/{_name}.txt"
            df = pd.read_csv(url, sep="|")
            df = df.rename(columns={"ACT Symbol": "Symbol"})
            _symbols = df["Symbol"].dropna()
            _symbols = _symbols.str.replace("$", "-P", regex=False)
            _symbols = _symbols.str.replace(".W", "-WT", regex=False)
            _symbols = _symbols.str.replace(".U", "-UN", regex=False)
            _symbols = _symbols.str.replace(".R", "-RI", regex=False)
            _symbols = _symbols.str.replace(".", "-", regex=False)
            _res_symbols += _symbols.unique().tolist()
        return _res_symbols

    @deco_retry
    def _get_nyse():
        url = "https://www.nyse.com/api/quotes/filter"
        _parms = {
            "instrumentType": "EQUITY",
            "pageNumber": 1,
            "sortColumn": "NORMALIZED_TICKER",
            "sortOrder": "ASC",
            "maxResultsPerPage": 10000,
            "filterToken": "",
        }
        resp = requests.post(url, json=_parms, timeout=None)
        if resp.status_code != 200:
            raise ValueError("request error")

        try:
            _symbols = [_v["symbolTicker"].replace("-", "-P") for _v in resp.json()]
        except Exception as e:
            logger.warning(f"request error: {e}")
            _symbols = []
        return _symbols

    if _US_SYMBOLS is None:
        _all_symbols = _get_eastmoney() + _get_nasdaq() + _get_nyse()
        if qlib_data_path is not None:
            for _index in ["nasdaq100", "sp500"]:
                ins_df = pd.read_csv(
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                )
                _all_symbols += ins_df["symbol"].unique().tolist()

        def _format(s_):
            s_ = s_.replace(".", "-")
            s_ = s_.strip("$")
            s_ = s_.strip("*")
            return s_

        _US_SYMBOLS = sorted(set(map(_format, filter(lambda x: len(x) < 8 and not x.endswith("WS"), _all_symbols))))

    return _US_SYMBOLS


def get_in_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """获取印度股票代码

    返回值
    -------
        股票代码列表
    """
    global _IN_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_nifty():
        url = f"https://www1.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df = df.rename(columns={"SYMBOL": "Symbol"})
        df["Symbol"] = df["Symbol"] + ".NS"
        _symbols = df["Symbol"].dropna()
        _symbols = _symbols.unique().tolist()
        return _symbols

    if _IN_SYMBOLS is None:
        _all_symbols = _get_nifty()
        if qlib_data_path is not None:
            for _index in ["nifty"]:
                ins_df = pd.read_csv(
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                )
                _all_symbols += ins_df["symbol"].unique().tolist()

        def _format(s_):
            s_ = s_.replace(".", "-")
            s_ = s_.strip("$")
            s_ = s_.strip("*")
            return s_

        _IN_SYMBOLS = sorted(set(_all_symbols))

    return _IN_SYMBOLS


def get_br_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """获取巴西(B3)股票代码

    返回值
    -------
        B3股票代码列表
    """
    global _BR_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_ibovespa():
        _symbols = []
        url = "https://www.fundamentus.com.br/detalhes.php?papel="

        # Request
        agent = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=agent, timeout=None)

        # BeautifulSoup
        soup = BeautifulSoup(page.content, "html.parser")
        tbody = soup.find("tbody")

        children = tbody.findChildren("a", recursive=True)
        for child in children:
            _symbols.append(str(child).rsplit('"', maxsplit=1)[-1].split(">")[1].split("<")[0])

        return _symbols

    if _BR_SYMBOLS is None:
        _all_symbols = _get_ibovespa()
        if qlib_data_path is not None:
            for _index in ["ibov"]:
                ins_df = pd.read_csv(
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                )
                _all_symbols += ins_df["symbol"].unique().tolist()

        def _format(s_):
            s_ = s_.strip()
            s_ = s_.strip("$")
            s_ = s_.strip("*")
            s_ = s_ + ".SA"
            return s_

        _BR_SYMBOLS = sorted(set(map(_format, _all_symbols)))

    return _BR_SYMBOLS


def get_en_fund_symbols(qlib_data_path: [str, Path] = None) -> list:
    """获取英文基金代码

    返回值
    -------
        中国市场基金代码列表
    """
    global _EN_FUND_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_eastmoney():
        url = "http://fund.eastmoney.com/js/fundcode_search.js"
        resp = requests.get(url, timeout=None)
        if resp.status_code != 200:
            raise ValueError("request error")
        try:
            _symbols = []
            for sub_data in re.findall(r"[\[](.*?)[\]]", resp.content.decode().split("= [")[-1].replace("];", "")):
                data = sub_data.replace('"', "").replace("'", "")
                # TODO: do we need other information, like fund_name from ['000001', 'HXCZHH', '华夏成长混合', '混合型', 'HUAXIACHENGZHANGHUNHE']
                _symbols.append(data.split(",")[0])
        except Exception as e:
            logger.warning(f"request error: {e}")
            raise
        if len(_symbols) < 8000:
            raise ValueError("request error")
        return _symbols

    if _EN_FUND_SYMBOLS is None:
        _all_symbols = _get_eastmoney()

        _EN_FUND_SYMBOLS = sorted(set(_all_symbols))

    return _EN_FUND_SYMBOLS


def symbol_suffix_to_prefix(symbol: str, capital: bool = True) -> str:
    """将股票代码后缀转换为前缀

    参数说明
    ----------
    symbol: str
        股票代码
    capital : bool
        是否大写，默认为True

    返回值
    -------
        转换后的股票代码
    """
    code, exchange = symbol.split(".")
    if exchange.lower() in ["sh", "ss"]:
        res = f"sh{code}"
    else:
        res = f"{exchange}{code}"
    return res.upper() if capital else res.lower()


def symbol_prefix_to_sufix(symbol: str, capital: bool = True) -> str:
    """将股票代码前缀转换为后缀

    参数说明
    ----------
    symbol: str
        股票代码
    capital : bool
        是否大写，默认为True

    返回值
    -------
        转换后的股票代码
    """
    res = f"{symbol[:-2]}.{symbol[-2:]}"
    return res.upper() if capital else res.lower()


def deco_retry(retry: int = 5, retry_sleep: int = 3):
    def deco_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retry = 5 if callable(retry) else retry
            _result = None
            for _i in range(1, _retry + 1):
                try:
                    _result = func(*args, **kwargs)
                    break

                except Exception as e:
                    logger.warning(f"{func.__name__}: {_i} :{e}")
                    if _i == _retry:
                        raise

                time.sleep(retry_sleep)
            return _result

        return wrapper

    return deco_func(retry) if callable(retry) else deco_func


def get_trading_date_by_shift(trading_list: list, trading_date: pd.Timestamp, shift: int = 1):
    """通过偏移获取交易日

    参数说明
    ----------
    trading_list: list
        交易日历列表
    shift : int
        偏移量，默认为1

    trading_date : pd.Timestamp
        交易日
    返回值
    -------
        偏移后的交易日
    """
    trading_date = pd.Timestamp(trading_date)
    left_index = bisect.bisect_left(trading_list, trading_date)
    try:
        res = trading_list[left_index + shift]
    except IndexError:
        res = trading_date
    return res


def generate_minutes_calendar_from_daily(
    calendars: Iterable,
    freq: str = "1min",
    am_range: Tuple[str, str] = ("09:30:00", "11:29:00"),
    pm_range: Tuple[str, str] = ("13:00:00", "14:59:00"),
) -> pd.Index:
    """生成分钟级交易日历

    参数说明
    ----------
    calendars: Iterable
        日线交易日历
    freq: str
        时间频率，默认为1min
    am_range: Tuple[str, str]
        上午交易时段，默认为中国股市: ("09:30:00", "11:29:00")
    pm_range: Tuple[str, str]
        下午交易时段，默认为中国股市: ("13:00:00", "14:59:00")

    """
    daily_format: str = "%Y-%m-%d"
    res = []
    for _day in calendars:
        for _range in [am_range, pm_range]:
            res.append(
                pd.date_range(
                    f"{pd.Timestamp(_day).strftime(daily_format)} {_range[0]}",
                    f"{pd.Timestamp(_day).strftime(daily_format)} {_range[1]}",
                    freq=freq,
                )
            )

    return pd.Index(sorted(set(np.hstack(res))))


def get_instruments(
    qlib_dir: str,
    index_name: str,
    method: str = "parse_instruments",
    freq: str = "day",
    request_retry: int = 5,
    retry_sleep: int = 3,
    market_index: str = "cn_index",
):
    """

    参数说明
    ----------
    qlib_dir: str
        qlib数据目录，默认为"Path(__file__).parent/qlib_data"
    index_name: str
        指数名称，可选值为["csi100", "csi300"]
    method: str
        方法名称，可选值为["parse_instruments", "save_new_companies"]
    freq: str
        频率，可选值为["day", "1min"]
    request_retry: int
        请求重试次数，默认为5
    retry_sleep: int
        请求重试间隔时间(秒)，默认为3
    market_index: str
        获取指数文件的位置，例如data_collector.cn_index.collector

    示例
    -------
        # 解析工具
        $ python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method parse_instruments

        # 解析新公司
        $ python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method save_new_companies

    """
    _cur_module = importlib.import_module("data_collector.{}.collector".format(market_index))
    obj = getattr(_cur_module, f"{index_name.upper()}Index")(
        qlib_dir=qlib_dir, index_name=index_name, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
    )
    getattr(obj, method)()


def _get_all_1d_data(_date_field_name: str, _symbol_field_name: str, _1d_data_all: pd.DataFrame):
    df = copy.deepcopy(_1d_data_all)
    df.reset_index(inplace=True)
    df.rename(columns={"datetime": _date_field_name, "instrument": _symbol_field_name}, inplace=True)
    df.columns = list(map(lambda x: x[1:] if x.startswith("$") else x, df.columns))
    return df


def get_1d_data(
    _date_field_name: str,
    _symbol_field_name: str,
    symbol: str,
    start: str,
    end: str,
    _1d_data_all: pd.DataFrame,
) -> pd.DataFrame:
    """获取日线数据

    返回值
    ------
        data_1d: pd.DataFrame
            data_1d.columns = [_date_field_name, _symbol_field_name, "paused", "volume", "factor", "close"]

    """
    _all_1d_data = _get_all_1d_data(_date_field_name, _symbol_field_name, _1d_data_all)
    return _all_1d_data[
        (_all_1d_data[_symbol_field_name] == symbol.upper())
        & (_all_1d_data[_date_field_name] >= pd.Timestamp(start))
        & (_all_1d_data[_date_field_name] < pd.Timestamp(end))
    ]


def calc_adjusted_price(
    df: pd.DataFrame,
    _1d_data_all: pd.DataFrame,
    _date_field_name: str,
    _symbol_field_name: str,
    frequence: str,
    consistent_1d: bool = True,
    calc_paused: bool = True,
) -> pd.DataFrame:
    """计算复权价格
    此方法执行4个操作：
    1. 添加`paused`字段
        - 添加的paused字段来自日线数据的paused字段
    2. 对齐日线数据的时间
    3. 数据重新加权
        - 加权方法：
            - volume / factor
            - open * factor
            - high * factor
            - low * factor
            - close * factor
    4. 调用`calc_paused_num`方法添加`paused_num`字段
        - `paused_num`是连续停牌天数
    """
    # TODO: 使用日线数据因子
    if df.empty:
        return df
    df = df.copy()
    df.drop_duplicates(subset=_date_field_name, inplace=True)
    df.sort_values(_date_field_name, inplace=True)
    symbol = df.iloc[0][_symbol_field_name]
    df[_date_field_name] = pd.to_datetime(df[_date_field_name])
    # 从qlib获取日线数据
    _start = pd.Timestamp(df[_date_field_name].min()).strftime("%Y-%m-%d")
    _end = (pd.Timestamp(df[_date_field_name].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data_1d: pd.DataFrame = get_1d_data(_date_field_name, _symbol_field_name, symbol, _start, _end, _1d_data_all)
    data_1d = data_1d.copy()
    if data_1d is None or data_1d.empty:
        df["factor"] = 1 / df.loc[df["close"].first_valid_index()]["close"]
        # TODO: 使用np.nan还是1或0
        df["paused"] = np.nan
    else:
        # 注意: 当volume为np.nan或volume <= 0时，paused = 1
        # FIXME: 寻找更准确的数据源
        data_1d["paused"] = 0
        data_1d.loc[(data_1d["volume"].isna()) | (data_1d["volume"] <= 0), "paused"] = 1
        data_1d = data_1d.set_index(_date_field_name)

        # 从日线数据添加因子
        # 注意: 日线数据说明:
        #   - 复权收盘价已考虑拆股因素。调整后的收盘价同时考虑了股息和拆股因素。
        #   - data_1d.adjclose: 同时考虑股息和拆股因素的调整后收盘价。
        #   - data_1d.close: `data_1d.adjclose / (首个非np.nan交易日的收盘价)`
        def _calc_factor(df_1d: pd.DataFrame):
            try:
                _date = pd.Timestamp(pd.Timestamp(df_1d[_date_field_name].iloc[0]).date())
                df_1d["factor"] = data_1d.loc[_date]["close"] / df_1d.loc[df_1d["close"].last_valid_index()]["close"]
                df_1d["paused"] = data_1d.loc[_date]["paused"]
            except Exception:
                df_1d["factor"] = np.nan
                df_1d["paused"] = np.nan
            return df_1d

        df = df.groupby([df[_date_field_name].dt.date], group_keys=False).apply(_calc_factor)
        if consistent_1d:
            # 日期序列与日线数据保持一致
            df.set_index(_date_field_name, inplace=True)
            df = df.reindex(
                generate_minutes_calendar_from_daily(
                    calendars=pd.to_datetime(data_1d.reset_index()[_date_field_name].drop_duplicates()),
                    freq=frequence,
                    am_range=("09:30:00", "11:29:00"),
                    pm_range=("13:00:00", "14:59:00"),
                )
            )
            df[_symbol_field_name] = df.loc[df[_symbol_field_name].first_valid_index()][_symbol_field_name]
            df.index.names = [_date_field_name]
            df.reset_index(inplace=True)
    for _col in ["open", "close", "high", "low", "volume"]:
        if _col not in df.columns:
            continue
        if _col == "volume":
            df[_col] = df[_col] / df["factor"]
        else:
            df[_col] = df[_col] * df["factor"]
    if calc_paused:
        df = calc_paused_num(df, _date_field_name, _symbol_field_name)
    return df


def calc_paused_num(df: pd.DataFrame, _date_field_name, _symbol_field_name):
    """计算停牌天数
    此方法添加paused_num字段
        - `paused_num`是连续停牌的天数
    """
    _symbol = df.iloc[0][_symbol_field_name]
    df = df.copy()
    df["_tmp_date"] = df[_date_field_name].apply(lambda x: pd.Timestamp(x).date())
    # 移除全天数据均为`np.nan`的起始和结束部分
    all_data = []
    # 记录全天为nan的连续交易日数量，用于移除最后一个全天为nan的交易日
    all_nan_nums = 0
    # 记录全天非nan的交易日连续出现次数
    not_nan_nums = 0
    for _date, _df in df.groupby("_tmp_date", group_keys=False):
        _df["paused"] = 0
        if not _df.loc[_df["volume"] < 0].empty:
            logger.warning(f"volume < 0, will fill np.nan: {_date} {_symbol}")
            _df.loc[_df["volume"] < 0, "volume"] = np.nan

        check_fields = set(_df.columns) - {
            "_tmp_date",
            "paused",
            "factor",
            _date_field_name,
            _symbol_field_name,
        }
        if _df.loc[:, list(check_fields)].isna().values.all() or (_df["volume"] == 0).all():
            all_nan_nums += 1
            not_nan_nums = 0
            _df["paused"] = 1
            if all_data:
                _df["paused_num"] = not_nan_nums
                all_data.append(_df)
        else:
            all_nan_nums = 0
            not_nan_nums += 1
            _df["paused_num"] = not_nan_nums
            all_data.append(_df)
    all_data = all_data[: len(all_data) - all_nan_nums]
    if all_data:
        df = pd.concat(all_data, sort=False)
    else:
        logger.warning(f"data is empty: {_symbol}")
        df = pd.DataFrame()
        return df
    del df["_tmp_date"]
    return df


if __name__ == "__main__":
    assert len(get_hs_stock_symbols()) >= MINIMUM_SYMBOLS_NUM
