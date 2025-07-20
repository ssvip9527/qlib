# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
from abc import abstractmethod
import warnings
from typing import Callable, Union, Tuple, List, Iterator, Optional

import pandas as pd

from qlib.typehint import Literal
from ...log import get_module_logger, TimeInspector
from ...utils import init_instance_by_config
from ...utils.serial import Serializable
from .utils import fetch_df_by_index, fetch_df_by_col
from ...utils import lazy_sort_index
from .loader import DataLoader

from . import processor as processor_module
from . import loader as data_loader_module


DATA_KEY_TYPE = Literal["raw", "infer", "learn"]


class DataHandlerABC(Serializable):
    """
    数据处理器接口。

    此类不假设数据处理器的内部数据结构。
    它仅为外部用户定义接口（使用DataFrame作为内部数据结构）。

    未来，数据处理器的更详细实现应进行重构。以下是一些指导原则：

    它包含几个组件：

    - [数据加载器] -> 数据的内部表示 -> 数据预处理 -> 获取接口的适配器
    - 组合所有组件的工作流程：
      工作流程可能非常复杂。DataHandlerLP是其中一种实践，但无法满足所有需求。
      因此，为用户提供实现工作流程的灵活性是更合理的选择。
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=W0246
        """
        我们应该定义如何为数据获取做好准备。
        """
        super().__init__(*args, **kwargs)

    CS_ALL = "__all"  # 返回具有单级索引列的所有列
    CS_RAW = "__raw"  # 返回具有多级索引列的原始数据

    # data key
    DK_R: DATA_KEY_TYPE = "raw"
    DK_I: DATA_KEY_TYPE = "infer"
    DK_L: DATA_KEY_TYPE = "learn"

    @abstractmethod
    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = CS_ALL,
        data_key: DATA_KEY_TYPE = DK_I,
    ) -> pd.DataFrame:
        pass


class DataHandler(DataHandlerABC):
    """
    DataHandler的设计动机：

    - 它提供了BaseDataHandler的一种实现，具体包括：
        - 使用内部加载的DataFrame处理响应
        - DataFrame由数据加载器加载

    使用处理器的步骤：
    1. 初始化数据处理器（通过`init`调用）。
    2. 使用数据。


    数据处理器尝试维护具有两级索引的处理器：
    `datetime`（日期时间）和`instruments`（标的）。

    支持任何顺序的索引级别（顺序将由数据暗示）。
    当数据框索引名称缺失时，将使用<`datetime`, `instruments`>顺序。

    数据示例：
    列的多级索引是可选的。

    .. code-block:: text

                                feature                                                            label
                                $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low  LABEL0
        datetime   instrument
        2010-01-04 SH600000    81.807068  17145150.0       83.737389        83.016739    2.741058  0.0032
                   SH600004    13.313329  11800983.0       13.313329        13.317701    0.183632  0.0042
                   SH600005    37.796539  12231662.0       38.258602        37.919757    0.970325  0.0289


    Tips for improving the performance of datahandler
    - Fetching data with `col_set=CS_RAW` will return the raw data and may avoid pandas from copying the data when calling `loc`
    """

    _data: pd.DataFrame  # underlying data.

    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader: Union[dict, str, DataLoader] = None,
        init_data=True,
        fetch_orig=True,
    ):
        """
        :param instruments: 要检索的股票列表
        :param start_time: 原始数据的开始时间
        :param end_time: 原始数据的结束时间
        :param data_loader: 用于加载数据的数据加载器
        :type data_loader: Union[dict, str, DataLoader]
        :param init_data: 在构造函数中初始化原始数据
        :param fetch_orig: 如果可能，返回原始数据而不是副本
        :type fetch_orig: bool
        """

        # Setup data loader
        assert data_loader is not None  # to make start_time end_time could have None default value

        # what data source to load data
        self.data_loader = init_instance_by_config(
            data_loader,
            None if (isinstance(data_loader, dict) and "module_path" in data_loader) else data_loader_module,
            accept_types=DataLoader,
        )

        # what data to be loaded from data source
        # For IDE auto-completion.
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time

        self.fetch_orig = fetch_orig
        if init_data:
            with TimeInspector.logt("Init data"):
                self.setup_data()
        super().__init__()

    def config(self, **kwargs):
        """
        数据配置。
        # 从数据源加载哪些数据

        此方法将在从数据集加载序列化的处理器时使用。
        数据将使用不同的时间范围进行初始化。
        """
        attr_list = {"instruments", "start_time", "end_time"}
        for k, v in kwargs.items():
            if k in attr_list:
                setattr(self, k, v)

        for attr in attr_list:
            if attr in kwargs:
                kwargs.pop(attr)

        super().config(**kwargs)

    def setup_data(self, enable_cache: bool = False):
        """
        设置数据，以防多次运行初始化

        负责维护以下变量：
        1) self._data

        参数
        ----------
        enable_cache : bool
            默认值为false：

            - 如果`enable_cache` == True：
                处理后的数据将保存到磁盘，下次调用`init`时处理器将直接从磁盘加载缓存数据
        """
        # 设置数据。
        # _data可能具有多级列索引。外层级别表示特征集名称
        with TimeInspector.logt("Loading data"):
            # 确保fetch方法基于索引排序的pd.DataFrame
            self._data = lazy_sort_index(self.data_loader.load(self.instruments, self.start_time, self.end_time))
        # TODO: cache

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandlerABC.CS_ALL,
        data_key: DATA_KEY_TYPE = DataHandlerABC.DK_I,
        squeeze: bool = False,
        proc_func: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        从底层数据源获取数据

        设计动机：
        - 为底层数据提供统一接口
        - 潜在地使接口更友好
        - 用户可以在此额外层中提高数据获取性能

        参数
        ----------
        selector : Union[pd.Timestamp, slice, str]
            描述如何按索引选择数据
            可以分为以下几类：

            - 获取单个索引
            - 获取索引范围

                - 切片范围
                - 特定索引的pd.Index

            可能出现以下冲突：

            - ["20200101", "20210101"]是表示选择此切片还是这两天？

                - 切片具有更高优先级

        level : Union[str, int]
            选择数据的索引级别

        col_set : Union[str, List[str]]

            - 如果是str类型：

                选择一组有意义的pd.Index列（例如特征、列）

                - 如果col_set == CS_RAW：

                    将返回原始数据集

            - 如果是List[str]类型：

                选择几组有意义的列，返回的数据具有多级索引

        proc_func: Callable

            - 提供在获取数据前处理数据的钩子
            - 解释此钩子必要性的示例：

                - 数据集学习了一些与数据分割相关的处理器
                - 每次准备数据时都会应用它们
                - 学习到的处理器要求数据框在拟合和应用时保持相同格式
                - 然而数据格式会根据参数变化
                - 因此处理器应应用于底层数据

        squeeze : bool
            是否压缩列和索引

        返回
        -------
        pd.DataFrame.
        """
        # DataHandler is an example with only one dataframe, so data_key is not used.
        _ = data_key  # avoid linting errors (e.g., unused-argument)
        return self._fetch_data(
            data_storage=self._data,
            selector=selector,
            level=level,
            col_set=col_set,
            squeeze=squeeze,
            proc_func=proc_func,
        )

    def _fetch_data(
        self,
        data_storage,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandlerABC.CS_ALL,
        squeeze: bool = False,
        proc_func: Callable = None,
    ):
        # 此方法被提取出来供子类共享
        from .storage import BaseHandlerStorage  # pylint: disable=C0415

        # 可能出现以下冲突
        # - ["20200101", "20210101"]是表示选择此切片还是这两天？
        # 为解决此问题
        #   - 切片具有更高优先级（除非level为None）
        if isinstance(selector, (tuple, list)) and level is not None:
            # 当level为None时，参数将直接传入
            # 无需转换为切片
            try:
                selector = slice(*selector)
            except ValueError:
                get_module_logger("DataHandlerLP").info(f"无法将查询转换为切片。将直接使用")

        if isinstance(data_storage, pd.DataFrame):
            data_df = data_storage
            if proc_func is not None:
                # FIXME: fetching by time first will be more friendly to `proc_func`
                # Copy in case of `proc_func` changing the data inplace....
                data_df = proc_func(fetch_df_by_index(data_df, selector, level, fetch_orig=self.fetch_orig).copy())
                data_df = fetch_df_by_col(data_df, col_set)
            else:
                # Fetch column  first will be more friendly to SepDataFrame
                data_df = fetch_df_by_col(data_df, col_set)
                data_df = fetch_df_by_index(data_df, selector, level, fetch_orig=self.fetch_orig)
        elif isinstance(data_storage, BaseHandlerStorage):
            if proc_func is not None:
                raise ValueError(f"proc_func is not supported by the storage {type(data_storage)}")
            data_df = data_storage.fetch(selector=selector, level=level, col_set=col_set, fetch_orig=self.fetch_orig)
        else:
            raise TypeError(f"data_storage should be pd.DataFrame|HashingStockStorage, not {type(data_storage)}")

        if squeeze:
            # squeeze columns
            data_df = data_df.squeeze()
            # squeeze index
            if isinstance(selector, (str, pd.Timestamp)):
                data_df = data_df.reset_index(level=level, drop=True)
        return data_df

    def get_cols(self, col_set=DataHandlerABC.CS_ALL) -> list:
        """
        get the column names

        Parameters
        ----------
        col_set : str
            select a set of meaningful columns.(e.g. features, columns)

        Returns
        -------
        list:
            list of column names
        """
        df = self._data.head()
        df = fetch_df_by_col(df, col_set)
        return df.columns.to_list()

    def get_range_selector(self, cur_date: Union[pd.Timestamp, str], periods: int) -> slice:
        """
        get range selector by number of periods

        Args:
            cur_date (pd.Timestamp or str): current date
            periods (int): number of periods
        """
        trading_dates = self._data.index.unique(level="datetime")
        cur_loc = trading_dates.get_loc(cur_date)
        pre_loc = cur_loc - periods + 1
        if pre_loc < 0:
            warnings.warn("`periods` is too large. the first date will be returned.")
            pre_loc = 0
        ref_date = trading_dates[pre_loc]
        return slice(ref_date, cur_date)

    def get_range_iterator(
        self, periods: int, min_periods: Optional[int] = None, **kwargs
    ) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
        """
        get an iterator of sliced data with given periods

        Args:
            periods (int): number of periods.
            min_periods (int): minimum periods for sliced dataframe.
            kwargs (dict): will be passed to `self.fetch`.
        """
        trading_dates = self._data.index.unique(level="datetime")
        if min_periods is None:
            min_periods = periods
        for cur_date in trading_dates[min_periods:]:
            selector = self.get_range_selector(cur_date, periods)
            yield cur_date, self.fetch(selector, **kwargs)


class DataHandlerLP(DataHandler):
    """
    Motivation:
    - For the case that we hope using different processor workflows for learning and inference;


    DataHandler with **(L)earnable (P)rocessor**

    This handler will produce three pieces of data in pd.DataFrame format.

    - DK_R / self._data: the raw data loaded from the loader
    - DK_I / self._infer: the data processed for inference
    - DK_L / self._learn: the data processed for learning model.

    The motivation of using different processor workflows for learning and inference
    Here are some examples.

    - The instrument universe for learning and inference may be different.
    - The processing of some samples may rely on label (for example, some samples hit the limit may need extra processing or be dropped).

        - These processors only apply to the learning phase.

    Tips for data handler

    - To reduce the memory cost

        - `drop_raw=True`: this will modify the data inplace on raw data;

    - Please note processed data like `self._infer` or `self._learn` are concepts different from `segments` in Qlib's `Dataset` like "train" and "test"

        - Processed data like `self._infer` or `self._learn` are underlying data processed with different processors
        - `segments` in Qlib's `Dataset` like "train" and "test" are simply the time segmentations when querying data("train" are often before "test" in time-series).
        - For example, you can query `data._infer` processed by `infer_processors` in the "train" time segmentation.
    """

    # based on `self._data`, _infer and _learn are genrated after processors
    _infer: pd.DataFrame  # data for inference
    _learn: pd.DataFrame  # data for learning models

    # map data_key to attribute name
    ATTR_MAP = {DataHandler.DK_R: "_data", DataHandler.DK_I: "_infer", DataHandler.DK_L: "_learn"}

    # process type
    PTYPE_I = "independent"
    # - self._infer will be processed by shared_processors + infer_processors
    # - self._learn will be processed by shared_processors + learn_processors

    # NOTE:
    PTYPE_A = "append"

    # - self._infer will be processed by shared_processors + infer_processors
    # - self._learn will be processed by shared_processors + infer_processors + learn_processors
    #   - (e.g. self._infer processed by learn_processors )

    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader: Union[dict, str, DataLoader] = None,
        infer_processors: List = [],
        learn_processors: List = [],
        shared_processors: List = [],
        process_type=PTYPE_A,
        drop_raw=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        infer_processors : list
            - list of <description info> of processors to generate data for inference

            - example of <description info>:

            .. code-block::

                1) classname & kwargs:
                    {
                        "class": "MinMaxNorm",
                        "kwargs": {
                            "fit_start_time": "20080101",
                            "fit_end_time": "20121231"
                        }
                    }
                2) Only classname:
                    "DropnaFeature"
                3) object instance of Processor

        learn_processors : list
            similar to infer_processors, but for generating data for learning models

        process_type: str
            PTYPE_I = 'independent'

            - self._infer will be processed by infer_processors

            - self._learn will be processed by learn_processors

            PTYPE_A = 'append'

            - self._infer will be processed by infer_processors

            - self._learn will be processed by infer_processors + learn_processors

              - (e.g. self._infer processed by learn_processors )
        drop_raw: bool
            Whether to drop the raw data
        """

        # Setup preprocessor
        self.infer_processors = []  # for lint
        self.learn_processors = []  # for lint
        self.shared_processors = []  # for lint
        for pname in "infer_processors", "learn_processors", "shared_processors":
            for proc in locals()[pname]:
                getattr(self, pname).append(
                    init_instance_by_config(
                        proc,
                        None if (isinstance(proc, dict) and "module_path" in proc) else processor_module,
                        accept_types=processor_module.Processor,
                    )
                )

        self.process_type = process_type
        self.drop_raw = drop_raw
        super().__init__(instruments, start_time, end_time, data_loader, **kwargs)

    def get_all_processors(self):
        return self.shared_processors + self.infer_processors + self.learn_processors

    def fit(self):
        """
        fit data without processing the data
        """
        for proc in self.get_all_processors():
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                proc.fit(self._data)

    def fit_process_data(self):
        """
        fit and process data

        The input of the `fit` will be the output of the previous processor
        """
        self.process_data(with_fit=True)

    @staticmethod
    def _run_proc_l(
        df: pd.DataFrame, proc_l: List[processor_module.Processor], with_fit: bool, check_for_infer: bool
    ) -> pd.DataFrame:
        for proc in proc_l:
            if check_for_infer and not proc.is_for_infer():
                raise TypeError("Only processors usable for inference can be used in `infer_processors` ")
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                if with_fit:
                    proc.fit(df)
                df = proc(df)
        return df

    @staticmethod
    def _is_proc_readonly(proc_l: List[processor_module.Processor]):
        """
        NOTE: it will return True if `len(proc_l) == 0`
        """
        for p in proc_l:
            if not p.readonly():
                return False
        return True

    def process_data(self, with_fit: bool = False):
        """
        process_data data. Fun `processor.fit` if necessary

        Notation: (data)  [processor]

        # data processing flow of self.process_type == DataHandlerLP.PTYPE_I

        .. code-block:: text

            (self._data)-[shared_processors]-(_shared_df)-[learn_processors]-(_learn_df)
                                                   \\
                                                    -[infer_processors]-(_infer_df)

        # data processing flow of self.process_type == DataHandlerLP.PTYPE_A

        .. code-block:: text

            (self._data)-[shared_processors]-(_shared_df)-[infer_processors]-(_infer_df)-[learn_processors]-(_learn_df)

        Parameters
        ----------
        with_fit : bool
            The input of the `fit` will be the output of the previous processor
        """
        # shared data processors
        # 1) assign
        _shared_df = self._data
        if not self._is_proc_readonly(self.shared_processors):  # avoid modifying the original data
            _shared_df = _shared_df.copy()
        # 2) process
        _shared_df = self._run_proc_l(_shared_df, self.shared_processors, with_fit=with_fit, check_for_infer=True)

        # data for inference
        # 1) assign
        _infer_df = _shared_df
        if not self._is_proc_readonly(self.infer_processors):  # avoid modifying the original data
            _infer_df = _infer_df.copy()
        # 2) process
        _infer_df = self._run_proc_l(_infer_df, self.infer_processors, with_fit=with_fit, check_for_infer=True)

        self._infer = _infer_df

        # data for learning
        # 1) assign
        if self.process_type == DataHandlerLP.PTYPE_I:
            _learn_df = _shared_df
        elif self.process_type == DataHandlerLP.PTYPE_A:
            # based on `infer_df` and append the processor
            _learn_df = _infer_df
        else:
            raise NotImplementedError(f"This type of input is not supported")
        if not self._is_proc_readonly(self.learn_processors):  # avoid modifying the original  data
            _learn_df = _learn_df.copy()
        # 2) process
        _learn_df = self._run_proc_l(_learn_df, self.learn_processors, with_fit=with_fit, check_for_infer=False)

        self._learn = _learn_df

        if self.drop_raw:
            del self._data

    def config(self, processor_kwargs: dict = None, **kwargs):
        """
        configuration of data.
        # what data to be loaded from data source

        This method will be used when loading pickled handler from dataset.
        The data will be initialized with different time range.

        """
        super().config(**kwargs)
        if processor_kwargs is not None:
            for processor in self.get_all_processors():
                processor.config(**processor_kwargs)

    # init type
    IT_FIT_SEQ = "fit_seq"  # the input of `fit` will be the output of the previous processor
    IT_FIT_IND = "fit_ind"  # the input of `fit` will be the original df
    IT_LS = "load_state"  # The state of the object has been load by pickle

    def setup_data(self, init_type: str = IT_FIT_SEQ, **kwargs):
        """
        Set up the data in case of running initialization for multiple time

        Parameters
        ----------
        init_type : str
            The type `IT_*` listed above.
        enable_cache : bool
            default value is false:

            - if `enable_cache` == True:

                the processed data will be saved on disk, and handler will load the cached data from the disk directly
                when we call `init` next time
        """
        # init raw data
        super().setup_data(**kwargs)

        with TimeInspector.logt("fit & process data"):
            if init_type == DataHandlerLP.IT_FIT_IND:
                self.fit()
                self.process_data()
            elif init_type == DataHandlerLP.IT_LS:
                self.process_data()
            elif init_type == DataHandlerLP.IT_FIT_SEQ:
                self.fit_process_data()
            else:
                raise NotImplementedError(f"This type of input is not supported")

        # TODO: Be able to cache handler data. Save the memory for data processing

    def _get_df_by_key(self, data_key: DATA_KEY_TYPE = DataHandlerABC.DK_I) -> pd.DataFrame:
        if data_key == self.DK_R and self.drop_raw:
            raise AttributeError(
                "DataHandlerLP has not attribute _data, please set drop_raw = False if you want to use raw data"
            )
        df = getattr(self, self.ATTR_MAP[data_key])
        return df

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set=DataHandler.CS_ALL,
        data_key: DATA_KEY_TYPE = DataHandler.DK_I,
        squeeze: bool = False,
        proc_func: Callable = None,
    ) -> pd.DataFrame:
        """
        fetch data from underlying data source

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index.
        level : Union[str, int]
            which index level to select the data.
        col_set : str
            select a set of meaningful columns.(e.g. features, columns).
        data_key : str
            the data to fetch:  DK_*.
        proc_func: Callable
            please refer to the doc of DataHandler.fetch

        Returns
        -------
        pd.DataFrame:
        """

        return self._fetch_data(
            data_storage=self._get_df_by_key(data_key),
            selector=selector,
            level=level,
            col_set=col_set,
            squeeze=squeeze,
            proc_func=proc_func,
        )

    def get_cols(self, col_set=DataHandler.CS_ALL, data_key: DATA_KEY_TYPE = DataHandlerABC.DK_I) -> list:
        """
        get the column names

        Parameters
        ----------
        col_set : str
            select a set of meaningful columns.(e.g. features, columns).
        data_key : DATA_KEY_TYPE
            the data to fetch:  DK_*.

        Returns
        -------
        list:
            list of column names
        """
        df = self._get_df_by_key(data_key).head()
        df = fetch_df_by_col(df, col_set)
        return df.columns.to_list()

    @classmethod
    def cast(cls, handler: "DataHandlerLP") -> "DataHandlerLP":
        """
        Motivation

        - A user creates a datahandler in his customized package. Then he wants to share the processed handler to
          other users without introduce the package dependency and complicated data processing logic.
        - This class make it possible by casting the class to DataHandlerLP and only keep the processed data

        Parameters
        ----------
        handler : DataHandlerLP
            A subclass of DataHandlerLP

        Returns
        -------
        DataHandlerLP:
            the converted processed data
        """
        new_hd: DataHandlerLP = object.__new__(DataHandlerLP)
        new_hd.from_cast = True  # add a mark for the cast instance

        for key in list(DataHandlerLP.ATTR_MAP.values()) + [
            "instruments",
            "start_time",
            "end_time",
            "fetch_orig",
            "drop_raw",
        ]:
            setattr(new_hd, key, getattr(handler, key, None))
        return new_hd

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DataHandlerLP":
        """
        Motivation:
        - When user want to get a quick data handler.

        The created data handler will have only one shared Dataframe without processors.
        After creating the handler, user may often want to dump the handler for reuse
        Here is a typical use case

        .. code-block:: python

            from qlib.data.dataset import DataHandlerLP
            dh = DataHandlerLP.from_df(df)
            dh.to_pickle(fname, dump_all=True)

        TODO:
        - The StaticDataLoader is quite slow. It don't have to copy the data again...

        """
        loader = data_loader_module.StaticDataLoader(df)
        return cls(data_loader=loader)
