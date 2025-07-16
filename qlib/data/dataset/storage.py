from abc import abstractmethod
import pandas as pd
import numpy as np

from .handler import DataHandler
from typing import Union, List
from qlib.log import get_module_logger

from .utils import get_level_index, fetch_df_by_index, fetch_df_by_col


class BaseHandlerStorage:
    """
    数据处理器的基础数据存储
    - pd.DataFrame是Qlib数据处理器中的默认数据存储格式
    - 如果用户想要使用自定义数据存储，应定义继承自BaseHandlerStorage的子类，并实现以下方法
    """

    @abstractmethod
    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
    ) -> pd.DataFrame:
        """从数据存储中获取数据

        参数
        ----------
        selector : Union[pd.Timestamp, slice, str, pd.Index]
            描述如何通过索引选择数据
        level : Union[str, int]
            要选择数据的索引级别
            - 如果level为None，直接将selector应用于df
        col_set : Union[str, List[str]]
            - 如果是str类型：
                选择一组有意义的列（例如特征、标签列）
                如果col_set == DataHandler.CS_RAW：
                    返回原始数据集
            - 如果是List[str]类型：
                选择多组有意义的列，返回的数据具有多级列
        fetch_orig : bool
            如果可能，返回原始数据而非副本

        返回
        -------
        pd.DataFrame
            获取到的数据框
        """
        raise NotImplementedError("fetch is method not implemented!")


class NaiveDFStorage(BaseHandlerStorage):
    """数据处理器的Naive数据存储
    - NaiveDFStorage是数据处理器的简单数据存储
    - NaiveDFStorage接收pandas.DataFrame作为输入，并提供数据获取的接口支持
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
    ) -> pd.DataFrame:
        # 可能会出现以下冲突
        # - ["20200101", "20210101"]是表示选择这个切片还是这两天？
        # 为解决此问题
        # - 切片具有更高优先级（当level为None时除外）
        if isinstance(selector, (tuple, list)) and level is not None:
            # when level is None, the argument will be passed in directly
            # we don't have to convert it into slice
            try:
                selector = slice(*selector)
            except ValueError:
                get_module_logger("DataHandlerLP").info(f"Fail to converting to query to slice. It will used directly")

        data_df = self.df
        data_df = fetch_df_by_col(data_df, col_set)
        data_df = fetch_df_by_index(data_df, selector, level, fetch_orig=fetch_orig)
        return data_df


class HashingStockStorage(BaseHandlerStorage):
    """数据处理器的Hashing数据存储
    - 默认数据存储pandas.DataFrame在随机访问单只股票数据时速度较慢
    - HashingStockStorage通过`stock_id`键对多只股票的数据(pandas.DataFrame)进行哈希处理
    - HashingStockStorage将pandas.DataFrame哈希为一个字典，其键为stock_id(字符串)，值为该股票的数据(pandas.DataFrame)，格式如下：
        {
            stock1_id: stock1_data,
            stock2_id: stock2_data,
            ...
            stockn_id: stockn_data,
        }
    - 通过`fetch`方法，用户可以以比默认数据存储低得多的时间成本访问任何股票数据
    """

    def __init__(self, df):
        self.hash_df = dict()
        self.stock_level = get_level_index(df, "instrument")
        for k, v in df.groupby(level="instrument", group_keys=False):
            self.hash_df[k] = v
        self.columns = df.columns

    @staticmethod
    def from_df(df):
        return HashingStockStorage(df)

    def _fetch_hash_df_by_stock(self, selector, level):
        """使用股票选择器获取数据

        参数
        ----------
        selector : Union[pd.Timestamp, slice, str]
            描述如何通过索引选择数据
        level : Union[str, int]
            要选择数据的索引级别
            - 如果level为None，直接将selector应用于df
            - `_fetch_hash_df_by_stock`将解析参数`selector`中的股票选择器

        返回
        -------
        Dict
            键为stock_id，值为股票数据的字典
        """

        stock_selector = slice(None)
        time_selector = slice(None)  # by default not filter by time.

        if level is None:
            # For directly applying.
            if isinstance(selector, tuple) and self.stock_level < len(selector):
                # full selector format
                stock_selector = selector[self.stock_level]
                time_selector = selector[1 - self.stock_level]
            elif isinstance(selector, (list, str)) and self.stock_level == 0:
                # only stock selector
                stock_selector = selector
        elif level in ("instrument", self.stock_level):
            if isinstance(selector, tuple):
                # 注意：股票级别选择器怎么会是元组？
                stock_selector = selector[0]
                raise TypeError(
                    "我忘记为什么会出现这种情况了。但我认为这没有意义，所以为此情况引发错误。"
                )
            elif isinstance(selector, (list, str)):
                stock_selector = selector

        if not isinstance(stock_selector, (list, str)) and stock_selector != slice(None):
            raise TypeError(f"股票选择器必须是str|list类型或slice(None)，而不是{stock_selector}")

        if stock_selector == slice(None):
            return self.hash_df, time_selector

        if isinstance(stock_selector, str):
            stock_selector = [stock_selector]

        select_dict = dict()
        for each_stock in sorted(stock_selector):
            if each_stock in self.hash_df:
                select_dict[each_stock] = self.hash_df[each_stock]
        return select_dict, time_selector

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
    ) -> pd.DataFrame:
        fetch_stock_df_list, time_selector = self._fetch_hash_df_by_stock(selector=selector, level=level)
        fetch_stock_df_list = list(fetch_stock_df_list.values())
        for _index, stock_df in enumerate(fetch_stock_df_list):
            fetch_col_df = fetch_df_by_col(df=stock_df, col_set=col_set)
            fetch_index_df = fetch_df_by_index(
                df=fetch_col_df, selector=time_selector, level="datetime", fetch_orig=fetch_orig
            )
            fetch_stock_df_list[_index] = fetch_index_df
        if len(fetch_stock_df_list) == 0:
            index_names = ("instrument", "datetime") if self.stock_level == 0 else ("datetime", "instrument")
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=index_names), columns=self.columns, dtype=np.float32
            )
        elif len(fetch_stock_df_list) == 1:
            return fetch_stock_df_list[0]
        else:
            return pd.concat(fetch_stock_df_list, sort=False, copy=~fetch_orig)
