# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
index_data的设计动机
- Pandas提供了许多用户友好的接口，但在单一工具中集成过多功能会带来额外开销，使其比numpy慢很多。
    有些用户只需要一个带索引的简单numpy数据框，不想要如此复杂的工具。
    这类用户正是`index_data`的目标用户

`index_data`尝试模仿pandas的行为(部分API会有所不同，因为我们力求更简单直观)，但不会牺牲性能。它提供基本的numpy数据和简单索引功能。如果用户调用可能影响性能的API，index_data会抛出错误。
"""

from __future__ import annotations

from typing import Dict, Tuple, Union, Callable, List
import bisect

import numpy as np
import pandas as pd


def concat(data_list: Union[SingleData], axis=0) -> MultiData:
    """按索引连接所有SingleData。
    TODO: 目前仅支持SingleData

    参数
    ----------
    data_list : List[SingleData]
        要连接的SingleData列表

    返回
    -------
    MultiData
        维度为2的MultiData
    """
    if axis == 0:
        raise NotImplementedError(f"please implement this func when axis == 0")
    elif axis == 1:
        # get all index and row
        all_index = set()
        for index_data in data_list:
            all_index = all_index | set(index_data.index)
        all_index = list(all_index)
        all_index.sort()
        all_index_map = dict(zip(all_index, range(len(all_index))))

        # concat all
        tmp_data = np.full((len(all_index), len(data_list)), np.nan)
        for data_id, index_data in enumerate(data_list):
            assert isinstance(index_data, SingleData)
            now_data_map = [all_index_map[index] for index in index_data.index]
            tmp_data[now_data_map, data_id] = index_data.data
        return MultiData(tmp_data, all_index)
    else:
        raise ValueError(f"axis must be 0 or 1")


def sum_by_index(data_list: Union[SingleData], new_index: list, fill_value=0) -> SingleData:
    """按新索引汇总所有SingleData

    参数
    ----------
    data_list : List[SingleData]
        要汇总的SingleData列表
    new_index : list
        新SingleData的索引
    fill_value : float
        填充缺失值或替换np.nan的值

    返回
    -------
    SingleData
        包含新索引和汇总值后的SingleData
    """
    data_list = [data.to_dict() for data in data_list]
    data_sum = {}
    for id in new_index:
        item_sum = 0
        for data in data_list:
            if id in data and not np.isnan(data[id]):
                item_sum += data[id]
            else:
                item_sum += fill_value
        data_sum[id] = item_sum
    return SingleData(data_sum)


class Index:
    """
    用于索引(行或列)

    读操作优先级高于其他操作。
    因此此类设计为**只读**方式共享查询数据。
    修改操作会生成新的Index实例。

    注意：当前索引存在以下限制
    - 不支持重复索引值(仅考虑首次出现的位置)
    - 不考虑索引顺序!!!! 因此当索引有序时，切片行为不会与pandas一致
    """

    def __init__(self, idx_list: Union[List, pd.Index, "Index", int]):
        self.idx_list: np.ndarray = None  # using array type for index list will make things easier
        if isinstance(idx_list, Index):
            # Fast read-only copy
            self.idx_list = idx_list.idx_list
            self.index_map = idx_list.index_map
            self._is_sorted = idx_list._is_sorted
        elif isinstance(idx_list, int):
            self.index_map = self.idx_list = np.arange(idx_list)
            self._is_sorted = True
        else:
            # Check if all elements in idx_list are of the same type
            if not all(isinstance(x, type(idx_list[0])) for x in idx_list):
                raise TypeError("All elements in idx_list must be of the same type")
            # Check if all elements in idx_list are of the same datetime64 precision
            if isinstance(idx_list[0], np.datetime64) and not all(x.dtype == idx_list[0].dtype for x in idx_list):
                raise TypeError("All elements in idx_list must be of the same datetime64 precision")
            self.idx_list = np.array(idx_list)
            # NOTE: only the first appearance is indexed
            self.index_map = dict(zip(self.idx_list, range(len(self))))
            self._is_sorted = False

    def __getitem__(self, i: int):
        return self.idx_list[i]

    def _convert_type(self, item):
        """

        After user creates indices with Type A, user may query data with other types with the same info.
            This method try to make type conversion and make query sane rather than raising KeyError strictly

        Parameters
        ----------
        item :
            The item to query index
        """

        if self.idx_list.dtype.type is np.datetime64:
            if isinstance(item, pd.Timestamp):
                # This happens often when creating index based on pandas.DatetimeIndex and query with pd.Timestamp
                return item.to_numpy().astype(self.idx_list.dtype)
            elif isinstance(item, np.datetime64):
                # This happens often when creating index based on np.datetime64 and query with another precision
                return item.astype(self.idx_list.dtype)
            # NOTE: It is hard to consider every case at first.
            # We just try to cover part of cases to make it more user-friendly
        return item

    def index(self, item) -> int:
        """
        根据索引值获取整数索引

        参数
        ----------
        item :
            要查询的项

        返回
        -------
        int:
            项的索引位置

        Raises
        ------
        KeyError:
            当查询项不存在时抛出
        """
        try:
            return self.index_map[self._convert_type(item)]
        except IndexError as index_e:
            raise KeyError(f"{item} can't be found in {self}") from index_e

    def __or__(self, other: "Index"):
        return Index(idx_list=list(set(self.idx_list) | set(other.idx_list)))

    def __eq__(self, other: "Index"):
        # NOTE:  np.nan is not supported in the index
        if self.idx_list.shape != other.idx_list.shape:
            return False
        return (self.idx_list == other.idx_list).all()

    def __len__(self):
        return len(self.idx_list)

    def is_sorted(self):
        return self._is_sorted

    def sort(self) -> Tuple["Index", np.ndarray]:
        """
        对索引进行排序

        返回
        -------
        Tuple["Index", np.ndarray]:
            排序后的Index和变化后的索引数组
        """
        sorted_idx = np.argsort(self.idx_list)
        idx = Index(self.idx_list[sorted_idx])
        idx._is_sorted = True
        return idx, sorted_idx

    def tolist(self):
        """return the index with the format of list."""
        return self.idx_list.tolist()


class LocIndexer:
    """
    `Indexer`的行为类似于Pandas中的`LocIndexer`

    读操作优先级高于其他操作。
    因此此类设计为只读方式共享查询数据。
    修改操作会生成新的Index实例。
    """

    def __init__(self, index_data: "IndexData", indices: List[Index], int_loc: bool = False):
        self._indices: List[Index] = indices
        self._bind_id = index_data  # bind index data
        self._int_loc = int_loc
        assert self._bind_id.data.ndim == len(self._indices)

    @staticmethod
    def proc_idx_l(indices: List[Union[List, pd.Index, Index]], data_shape: Tuple = None) -> List[Index]:
        """处理用户输入的索引并输出`Index`列表"""
        res = []
        for i, idx in enumerate(indices):
            res.append(Index(data_shape[i] if len(idx) == 0 else idx))
        return res

    def _slc_convert(self, index: Index, indexing: slice) -> slice:
        """
        将基于值的索引转换为基于整数的索引

        参数
        ----------
        index : Index
            索引数据
        indexing : slice
            用于索引的基于值的切片类型数据

        返回
        -------
        slice:
            基于整数的切片
        """
        if index.is_sorted():
            int_start = None if indexing.start is None else bisect.bisect_left(index, indexing.start)
            int_stop = None if indexing.stop is None else bisect.bisect_right(index, indexing.stop)
        else:
            int_start = None if indexing.start is None else index.index(indexing.start)
            int_stop = None if indexing.stop is None else index.index(indexing.stop) + 1
        return slice(int_start, int_stop)

    def __getitem__(self, indexing):
        """

        参数
        ----------
        indexing :
            数据查询

        异常
        ------
        KeyError:
            当查询非切片索引但不存在时抛出
        """
        # 1) convert slices to int loc
        if not isinstance(indexing, tuple):
            # NOTE: tuple is not supported for indexing
            indexing = (indexing,)

        # TODO: create a subclass for single value query
        assert len(indexing) <= len(self._indices)

        int_indexing = []
        for dim, index in enumerate(self._indices):
            if dim < len(indexing):
                _indexing = indexing[dim]
                if not self._int_loc:  # type converting is only necessary when it is not `iloc`
                    if isinstance(_indexing, slice):
                        _indexing = self._slc_convert(index, _indexing)
                    elif isinstance(_indexing, (IndexData, np.ndarray)):
                        if isinstance(_indexing, IndexData):
                            _indexing = _indexing.data
                        assert _indexing.ndim == 1
                        if _indexing.dtype != bool:
                            _indexing = np.array(list(index.index(i) for i in _indexing))
                    else:
                        _indexing = index.index(_indexing)
            else:
                # Default to select all when user input is not given
                _indexing = slice(None)
            int_indexing.append(_indexing)

        # 2) select data and index
        new_data = self._bind_id.data[tuple(int_indexing)]
        # return directly if it is scalar
        if new_data.ndim == 0:
            return new_data
        # otherwise we go on to the index part
        new_indices = [idx[indexing] for idx, indexing in zip(self._indices, int_indexing)]

        # 3) squash dimensions
        new_indices = [
            idx for idx in new_indices if isinstance(idx, np.ndarray) and idx.ndim > 0
        ]  # squash the zero dim indexing

        if new_data.ndim == 1:
            cls = SingleData
        elif new_data.ndim == 2:
            cls = MultiData
        else:
            raise ValueError("Not supported")
        return cls(new_data, *new_indices)


class BinaryOps:
    def __init__(self, method_name):
        self.method_name = method_name

    def __get__(self, obj, *args):
        # bind object
        self.obj = obj
        return self

    def __call__(self, other):
        self_data_method = getattr(self.obj.data, self.method_name)

        if isinstance(other, (int, float, np.number)):
            return self.obj.__class__(self_data_method(other), *self.obj.indices)
        elif isinstance(other, self.obj.__class__):
            other_aligned = self.obj._align_indices(other)
            return self.obj.__class__(self_data_method(other_aligned.data), *self.obj.indices)
        else:
            return NotImplemented


def index_data_ops_creator(*args, **kwargs):
    """
    用于自动生成索引数据操作的元类
    """
    for method_name in ["__add__", "__sub__", "__rsub__", "__mul__", "__truediv__", "__eq__", "__gt__", "__lt__"]:
        args[2][method_name] = BinaryOps(method_name=method_name)
    return type(*args)


class IndexData(metaclass=index_data_ops_creator):
    """
    SingleData和MultiData的基础数据结构

    注意:
    - 出于性能考虑，底层数据仅支持**np.floating**类型!!!
    - 基于np.floating的布尔类型也受支持。示例如下

    .. code-block:: python

        np.array([ np.nan]).any() -> True
        np.array([ np.nan]).all() -> True
        np.array([1. , 0.]).any() -> True
        np.array([1. , 0.]).all() -> False
    """

    loc_idx_cls = LocIndexer

    def __init__(self, data: np.ndarray, *indices: Union[List, pd.Index, Index]):
        self.data = data
        self.indices = indices

        # get the expected data shape
        # - The index has higher priority
        self.data = np.array(data)

        expected_dim = max(self.data.ndim, len(indices))

        data_shape = []
        for i in range(expected_dim):
            idx_l = indices[i] if len(indices) > i else []
            if len(idx_l) == 0:
                data_shape.append(self.data.shape[i])
            else:
                data_shape.append(len(idx_l))
        data_shape = tuple(data_shape)

        # broadcast the data to expected shape
        if self.data.shape != data_shape:
            self.data = np.broadcast_to(self.data, data_shape)

        self.data = self.data.astype(np.float64)
        # Please notice following cases when converting the type
        # - np.array([None, 1]).astype(np.float64) -> array([nan,  1.])

        # create index from user's index data.
        self.indices: List[Index] = self.loc_idx_cls.proc_idx_l(indices, data_shape)

        for dim in range(expected_dim):
            assert self.data.shape[dim] == len(self.indices[dim])

        self.ndim = expected_dim

    # indexing related methods
    @property
    def loc(self):
        return self.loc_idx_cls(index_data=self, indices=self.indices)

    @property
    def iloc(self):
        return self.loc_idx_cls(index_data=self, indices=self.indices, int_loc=True)

    @property
    def index(self):
        return self.indices[0]

    @property
    def columns(self):
        return self.indices[1]

    def __getitem__(self, args):
        # NOTE: this tries to behave like a numpy array to be compatible with numpy aggregating function like nansum and nanmean
        return self.iloc[args]

    def _align_indices(self, other: "IndexData") -> "IndexData":
        """
        在执行算术运算前将`other`的所有索引对齐到`self`
        此函数将返回新的IndexData而不是原地修改`other`的数据

        参数
        ----------
        other : "IndexData"
            需要修改索引的数据

        返回
        -------
        IndexData:
            索引已对齐到`self`的数据
        """
        raise NotImplementedError(f"please implement _align_indices func")

    def sort_index(self, axis=0, inplace=True):
        assert inplace, "Only support sorting inplace now"
        self.indices[axis], sorted_idx = self.indices[axis].sort()
        self.data = np.take(self.data, sorted_idx, axis=axis)

    # The code below could be simpler like methods in __getattribute__
    def __invert__(self):
        return self.__class__(~self.data.astype(bool), *self.indices)

    def abs(self):
        """获取数据绝对值(np.nan除外)"""
        tmp_data = np.absolute(self.data)
        return self.__class__(tmp_data, *self.indices)

    def replace(self, to_replace: Dict[np.number, np.number]):
        assert isinstance(to_replace, dict)
        tmp_data = self.data.copy()
        for num in to_replace:
            if num in tmp_data:
                tmp_data[self.data == num] = to_replace[num]
        return self.__class__(tmp_data, *self.indices)

    def apply(self, func: Callable):
        """对数据应用函数"""
        tmp_data = func(self.data)
        return self.__class__(tmp_data, *self.indices)

    def __len__(self):
        """数据长度

        返回
        -------
        int
            数据长度
        """
        return len(self.data)

    def sum(self, axis=None, dtype=None, out=None):
        assert out is None and dtype is None, "`out` is just for compatible with numpy's aggregating function"
        # FIXME: weird logic and not general
        if axis is None:
            return np.nansum(self.data)
        elif axis == 0:
            tmp_data = np.nansum(self.data, axis=0)
            return SingleData(tmp_data, self.columns)
        elif axis == 1:
            tmp_data = np.nansum(self.data, axis=1)
            return SingleData(tmp_data, self.index)
        else:
            raise ValueError(f"axis must be None, 0 or 1")

    def mean(self, axis=None, dtype=None, out=None):
        assert out is None and dtype is None, "`out` is just for compatible with numpy's aggregating function"
        # FIXME: weird logic and not general
        if axis is None:
            return np.nanmean(self.data)
        elif axis == 0:
            tmp_data = np.nanmean(self.data, axis=0)
            return SingleData(tmp_data, self.columns)
        elif axis == 1:
            tmp_data = np.nanmean(self.data, axis=1)
            return SingleData(tmp_data, self.index)
        else:
            raise ValueError(f"axis must be None, 0 or 1")

    def isna(self):
        return self.__class__(np.isnan(self.data), *self.indices)

    def fillna(self, value=0.0, inplace: bool = False):
        if inplace:
            self.data = np.nan_to_num(self.data, nan=value)
        else:
            return self.__class__(np.nan_to_num(self.data, nan=value), *self.indices)

    def count(self):
        return len(self.data[~np.isnan(self.data)])

    def all(self):
        if None in self.data:
            return self.data[self.data is not None].all()
        else:
            return self.data.all()

    @property
    def empty(self):
        return len(self.data) == 0

    @property
    def values(self):
        return self.data


class SingleData(IndexData):
    def __init__(
        self, data: Union[int, float, np.number, list, dict, pd.Series] = [], index: Union[List, pd.Index, Index] = []
    ):
        """索引与numpy数据的数据结构
        用于替代pd.Series以获得更高性能

        参数
        ----------
        data : Union[int, float, np.number, list, dict, pd.Series]
            输入数据
        index : Union[list, pd.Index]
            数据索引
            空列表表示自动填充索引到数据长度
        """
        # for special data type
        if isinstance(data, dict):
            assert len(index) == 0
            if len(data) > 0:
                index, data = zip(*data.items())
            else:
                index, data = [], []
        elif isinstance(data, pd.Series):
            assert len(index) == 0
            index, data = data.index, data.values
        elif isinstance(data, (int, float, np.number)):
            data = [data]
        super().__init__(data, index)
        assert self.ndim == 1

    def _align_indices(self, other):
        if self.index == other.index:
            return other
        elif set(self.index) == set(other.index):
            return other.reindex(self.index)
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def reindex(self, index: Index, fill_value=np.nan) -> SingleData:
        """重新索引数据并用np.nan填充缺失值

        参数
        ----------
        new_index : list
            新索引
        fill_value:
            索引缺失时的填充值

        返回
        -------
        SingleData
            重新索引后的数据
        """
        # TODO: This method can be more general
        if self.index == index:
            return self
        tmp_data = np.full(len(index), fill_value, dtype=np.float64)
        for index_id, index_item in enumerate(index):
            try:
                tmp_data[index_id] = self.loc[index_item]
            except KeyError:
                pass
        return SingleData(tmp_data, index)

    def add(self, other: SingleData, fill_value=0):
        # TODO: add and __add__ are a little confusing.
        # This could be a more general
        common_index = self.index | other.index
        common_index, _ = common_index.sort()
        tmp_data1 = self.reindex(common_index, fill_value)
        tmp_data2 = other.reindex(common_index, fill_value)
        return tmp_data1.fillna(fill_value) + tmp_data2.fillna(fill_value)

    def to_dict(self):
        """将SingleData转换为字典

        返回
        -------
        dict
            字典格式的数据
        """
        return dict(zip(self.index, self.data.tolist()))

    def to_series(self):
        return pd.Series(self.data, index=self.index)

    def __repr__(self) -> str:
        return str(pd.Series(self.data, index=self.index.tolist()))


class MultiData(IndexData):
    def __init__(
        self,
        data: Union[int, float, np.number, list] = [],
        index: Union[List, pd.Index, Index] = [],
        columns: Union[List, pd.Index, Index] = [],
    ):
        """索引与numpy数据的数据结构
        用于替代pd.DataFrame以获得更高性能

        参数
        ----------
        data : Union[list, np.ndarray]
            数据维度必须为2
        index : Union[List, pd.Index, Index]
            数据索引
        columns: Union[List, pd.Index, Index]
            数据列名
        """
        if isinstance(data, pd.DataFrame):
            index, columns, data = data.index, data.columns, data.values
        super().__init__(data, index, columns)
        assert self.ndim == 2

    def _align_indices(self, other):
        if self.indices == other.indices:
            return other
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def __repr__(self) -> str:
        return str(pd.DataFrame(self.data, index=self.index.tolist(), columns=self.columns.tolist()))
