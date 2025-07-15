# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import Iterable, overload, Tuple, List, Text, Union, Dict

import numpy as np
import pandas as pd
from qlib.log import get_module_logger

# calendar value type
CalVT = str

# instrument value
InstVT = List[Tuple[CalVT, CalVT]]
# instrument key
InstKT = Text

logger = get_module_logger("storage")

"""
If the user is only using it in `qlib`, you can customize Storage to implement only the following methods:

class UserCalendarStorage(CalendarStorage):

    @property
    def data(self) -> Iterable[CalVT]:
        '''get all data

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        '''
        raise NotImplementedError("Subclass of CalendarStorage must implement `data` method")


class UserInstrumentStorage(InstrumentStorage):

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        '''get all data

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        '''
        raise NotImplementedError("Subclass of InstrumentStorage must implement `data` method")


class UserFeatureStorage(FeatureStorage):

    def __getitem__(self, s: slice) -> pd.Series:
        '''x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]

        Returns
        -------
            pd.Series(values, index=pd.RangeIndex(start, len(values))

        Notes
        -------
        if data(storage) does not exist:
            if isinstance(i, int):
                return (None, None)
            if isinstance(i,  slice):
                # return empty pd.Series
                return pd.Series(dtype=np.float32)
        '''
        raise NotImplementedError(
            "Subclass of FeatureStorage must implement `__getitem__(s: slice)` method"
        )


"""


class BaseStorage:
    """存储基类，提供存储名称的基本实现"""
    @property
    def storage_name(self) -> str:
        """获取存储名称

        返回:
            str: 存储名称（小写）
        """
        return re.findall("[A-Z][^A-Z]*", self.__class__.__name__)[-2].lower()


class CalendarStorage(BaseStorage):
    """
    日历存储类，其方法行为与同名的List方法保持一致
    """

    def __init__(self, freq: str, future: bool, **kwargs):
        """初始化日历存储

        参数:
            freq: 频率字符串
            future: 是否包含未来数据
            **kwargs: 其他关键字参数
        """
        self.freq = freq
        self.future = future
        self.kwargs = kwargs

    @property
    def data(self) -> Iterable[CalVT]:
        """获取所有日历数据

        返回:
            Iterable[CalVT]: 日历数据的可迭代对象

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        """
        raise NotImplementedError("CalendarStorage的子类必须实现`data`方法")

    def clear(self) -> None:
        """清空日历存储数据

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("CalendarStorage的子类必须实现`clear`方法")

    def extend(self, iterable: Iterable[CalVT]) -> None:
        """扩展日历存储数据

        参数:
            iterable: 包含日历数据的可迭代对象

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("CalendarStorage的子类必须实现`extend`方法")

    def index(self, value: CalVT) -> int:
        """获取日历值的索引

        参数:
            value: 日历值

        返回:
            int: 日历值在存储中的索引

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("CalendarStorage的子类必须实现`index`方法")

    def insert(self, index: int, value: CalVT) -> None:
        """在指定位置插入日历值

        参数:
            index: 插入位置
            value: 要插入的日历值

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("CalendarStorage的子类必须实现`insert`方法")

    def remove(self, value: CalVT) -> None:
        raise NotImplementedError("CalendarStorage的子类必须实现`remove`方法")

    @overload
    def __setitem__(self, i: int, value: CalVT) -> None:
        """设置指定索引位置的日历值

        x.__setitem__(i, o) <==> (x[i] = o)

        参数:
            i: 索引位置
            value: 要设置的日历值
        """

    @overload
    def __setitem__(self, s: slice, value: Iterable[CalVT]) -> None:
        """设置切片范围内的日历值

        x.__setitem__(s, o) <==> (x[s] = o)

        参数:
            s: 切片对象
            value: 要设置的日历值可迭代对象
        """

    def __setitem__(self, i, value) -> None:
        raise NotImplementedError(
            "CalendarStorage的子类必须实现`__setitem__(i: int, o: CalVT)`/`__setitem__(s: slice, o: Iterable[CalVT])`方法"
        )

    @overload
    def __delitem__(self, i: int) -> None:
        """删除指定索引的日历数据

        x.__delitem__(i) <==> del x[i]
        """

    @overload
    def __delitem__(self, i: slice) -> None:
        """删除切片范围内的日历数据
        x.__delitem__(slice(start: int, stop: int, step: int)) <==> del x[start:stop:step]
        """

    def __delitem__(self, i) -> None:
        """
        删除指定索引或切片的日历数据

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        """
        raise NotImplementedError(
            "CalendarStorage的子类必须实现`__delitem__(i: int)`/`__delitem__(s: slice)`方法"
        )

    @overload
    def __getitem__(self, s: slice) -> Iterable[CalVT]:
        """获取切片范围内的日历数据

        x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]

        返回:
            Iterable[CalVT]: 日历数据的可迭代对象
        """

    @overload
    def __getitem__(self, i: int) -> CalVT:
        """获取指定索引的日历数据

        x.__getitem__(i) <==> x[i]

        返回:
            CalVT: 日历值
        """

    def __getitem__(self, i) -> CalVT:
        """
        获取指定索引或切片的日历数据

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        """
        raise NotImplementedError(
            "CalendarStorage的子类必须实现`__getitem__(i: int)`/`__getitem__(s: slice)`方法"
        )

    def __len__(self) -> int:
        """
        获取日历存储的长度

        返回:
            int: 日历数据的数量

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        """
        raise NotImplementedError("CalendarStorage的子类必须实现`__len__`方法")


class InstrumentStorage(BaseStorage):
    """证券工具存储类，用于管理证券工具的相关数据"""
    def __init__(self, market: str, freq: str, **kwargs):
        """初始化证券工具存储

        参数:
            market: 市场名称
            freq: 频率字符串
            **kwargs: 其他关键字参数
        """
        self.market = market
        self.freq = freq
        self.kwargs = kwargs

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        """获取所有证券工具数据

        返回:
            Dict[InstKT, InstVT]: 证券工具数据字典，键为工具代码，值为包含起止时间的列表

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`data`方法")

    def clear(self) -> None:
        """清空证券工具存储数据

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`clear`方法")

    def update(self, *args, **kwargs) -> None:
        """更新证券工具存储数据

        D.update([E, ]**F) -> None. 从映射/可迭代对象E和F更新D。

        注意:
        ------
            如果提供了E且E有.keys()方法，则执行: for k in E: D[k] = E[k]

            如果提供了E但E没有.keys()方法，则执行: for (k, v) in E: D[k] = v

            在上述两种情况下，之后都会执行: for k, v in F.items(): D[k] = v

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`update`方法")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        """设置指定证券工具的数据

        参数:
            k: 证券工具代码
            v: 证券工具的起止时间列表

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`__setitem__`方法")

    def __delitem__(self, k: InstKT) -> None:
        """删除指定证券工具的数据

        参数:
            k: 证券工具代码

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`__delitem__`方法")

    def __getitem__(self, k: InstKT) -> InstVT:
        """获取指定证券工具的数据

        x.__getitem__(k) <==> x[k]

        参数:
            k: 证券工具代码

        返回:
            InstVT: 证券工具的起止时间列表

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`__getitem__`方法")

    def __len__(self) -> int:
        """
        获取证券工具的数量

        返回:
            int: 证券工具的数量

        异常:
        ------
        ValueError
            如果数据(存储)不存在，则引发ValueError
        """
        raise NotImplementedError("InstrumentStorage的子类必须实现`__len__`方法")


class FeatureStorage(BaseStorage):
    """特征存储类，用于管理证券特征数据"""
    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        """初始化特征存储

        参数:
            instrument: 证券工具代码
            field: 特征字段名称
            freq: 频率字符串
            **kwargs: 其他关键字参数
        """
        self.instrument = instrument
        self.field = field
        self.freq = freq
        self.kwargs = kwargs

    @property
    def data(self) -> pd.Series:
        """获取所有特征数据

        返回:
            pd.Series: 特征数据序列

        注意:
        ------
        如果数据(存储)不存在，返回空的pd.Series: `return pd.Series(dtype=np.float32)`
        """
        raise NotImplementedError("FeatureStorage的子类必须实现`data`方法")

    @property
    def start_index(self) -> Union[int, None]:
        """获取特征存储的起始索引

        返回:
            Union[int, None]: 起始索引，如果数据不存在则返回None

        注意:
        -----
        如果数据(存储)不存在，返回None
        """
        raise NotImplementedError("FeatureStorage的子类必须实现`start_index`方法")

    @property
    def end_index(self) -> Union[int, None]:
        """获取特征存储的结束索引

        返回:
            Union[int, None]: 结束索引，如果数据不存在则返回None

        注意:
        -----
        数据范围的右索引（闭区间）

            下一个数据追加点为 `end_index + 1`

        如果数据(存储)不存在，返回None
        """
        raise NotImplementedError("FeatureStorage的子类必须实现`end_index`方法")

    def clear(self) -> None:
        """清空特征存储数据

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("FeatureStorage的子类必须实现`clear`方法")

    def write(self, data_array: Union[List, np.ndarray, Tuple], index: int = None):
        """将数据数组写入特征存储，从指定索引开始

        参数:
            data_array: 要写入的数据数组，可以是列表、numpy数组或元组
            index: 起始索引，如果为None则追加数据

        注意:
        ------
            如果index为None，则将data_array追加到特征数据末尾

            如果data_array长度为0，则直接返回

            如果(index - self.end_index) >= 1，则self[end_index+1: index]区间将填充np.nan

        示例:
        ---------
            .. code-block::

                特征数据:
                    3   4
                    4   5
                    5   6


            >>> self.write([6, 7], index=6)

                特征数据:
                    3   4
                    4   5
                    5   6
                    6   6
                    7   7

            >>> self.write([8], index=9)

                特征数据:
                    3   4
                    4   5
                    5   6
                    6   6
                    7   7
                    8   np.nan
                    9   8

            >>> self.write([1, np.nan], index=3)

                特征数据:
                    3   1
                    4   np.nan
                    5   6
                    6   6
                    7   7
                    8   np.nan
                    9   8

        异常:
        ------
        NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("FeatureStorage的子类必须实现`write`方法")

    def rebase(self, start_index: int = None, end_index: int = None):
        """重新设置特征存储的起始索引和结束索引

        参数:
            start_index: 新的起始索引，默认为None（使用当前起始索引）
            end_index: 新的结束索引，默认为None（使用当前结束索引）

        注意:
        ------
        start_index和end_index构成闭区间: [start_index, end_index]

        示例:
        ---------

            .. code-block::

                    特征数据:
                        3   4
                        4   5
                        5   6


                >>> self.rebase(start_index=4)

                    特征数据:
                        4   5
                        5   6

                >>> self.rebase(start_index=3)

                    特征数据:
                        3   np.nan
                        4   5
                        5   6

                >>> self.write([3], index=3)

                    特征数据:
                        3   3
                        4   5
                        5   6

                >>> self.rebase(end_index=4)

                    特征数据:
                        3   3
                        4   5

                >>> self.write([6, 7, 8], index=4)

                    特征数据:
                        3   3
                        4   6
                        5   7
                        6   8

                >>> self.rebase(start_index=4, end_index=5)

                    特征数据:
                        4   6
                        5   7

        异常:
        ------
        ValueError: 如果storage.start_index或storage.end_index为None（存储可能不存在）
        """
        storage_si = self.start_index
        storage_ei = self.end_index
        if storage_si is None or storage_ei is None:
            raise ValueError("storage.start_index或storage.end_index为None，存储可能不存在")

        start_index = storage_si if start_index is None else start_index
        end_index = storage_ei if end_index is None else end_index

        if start_index is None or end_index is None:
            logger.warning("both start_index and end_index are None, or storage does not exist; rebase is ignored")
            return

        if start_index < 0 or end_index < 0:
            logger.warning("start_index or end_index cannot be less than 0")
            return
        if start_index > end_index:
            logger.warning(
                f"start_index({start_index}) > end_index({end_index}), rebase is ignored; "
                f"if you need to clear the FeatureStorage, please execute: FeatureStorage.clear"
            )
            return

        if start_index <= storage_si:
            self.write([np.nan] * (storage_si - start_index), start_index)
        else:
            self.rewrite(self[start_index:].values, start_index)

        if end_index >= self.end_index:
            self.write([np.nan] * (end_index - self.end_index))
        else:
            self.rewrite(self[: end_index + 1].values, start_index)

    def rewrite(self, data: Union[List, np.ndarray, Tuple], index: int):
        """overwrite all data in FeatureStorage with data

        Parameters
        ----------
        data: Union[List, np.ndarray, Tuple]
            data
        index: int
            data start index
        """
        self.clear()
        self.write(data, index)

    @overload
    def __getitem__(self, s: slice) -> pd.Series:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]

        Returns
        -------
            pd.Series(values, index=pd.RangeIndex(start, len(values))
        """

    @overload
    def __getitem__(self, i: int) -> Tuple[int, float]:
        """x.__getitem__(y) <==> x[y]"""

    def __getitem__(self, i) -> Union[Tuple[int, float], pd.Series]:
        """x.__getitem__(y) <==> x[y]

        Notes
        -------
        if data(storage) does not exist:
            if isinstance(i, int):
                return (None, None)
            if isinstance(i,  slice):
                # return empty pd.Series
                return pd.Series(dtype=np.float32)
        """
        raise NotImplementedError(
            "Subclass of FeatureStorage must implement `__getitem__(i: int)`/`__getitem__(s: slice)` method"
        )

    def __len__(self) -> int:
        """

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError

        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `__len__`  method")
