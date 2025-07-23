# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import pickle
from pathlib import Path
import warnings
import pandas as pd

from typing import Tuple, Union, List, Dict

from qlib.data import D
from qlib.utils import load_dataset, init_instance_by_config, time_to_slc_point
from qlib.log import get_module_logger
from qlib.utils.serial import Serializable


class DataLoader(abc.ABC):
    """
    DataLoader用于从原始数据源加载原始数据。
    """

    @abc.abstractmethod
    def load(self, instruments, start_time=None, end_time=None) -> pd.DataFrame:
        """
        以pd.DataFrame格式加载数据。

        数据示例（列的多级索引是可选的）：

            .. code-block:: text

                                        feature                                                             label
                                        $close     $volume     Ref($close, 1)  Mean($close, 3)  $high-$low  LABEL0
                datetime    instrument
                2010-01-04  SH600000    81.807068  17145150.0       83.737389        83.016739    2.741058  0.0032
                            SH600004    13.313329  11800983.0       13.313329        13.317701    0.183632  0.0042
                            SH600005    37.796539  12231662.0       38.258602        37.919757    0.970325  0.0289


        参数
        ----------
        instruments : str或dict
            可以是市场名称或由InstrumentProvider生成的标的配置文件。
            如果instruments的值为None，则表示不进行过滤。
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。

        返回
        -------
        pd.DataFrame:
            从底层数据源加载的数据

        异常
        -----        
        KeyError:
            如果不支持标的过滤，将引发KeyError
        """


class DLWParser(DataLoader):
    """
    (D)ata(L)oader (W)ith (P)arser（带解析器的数据加载器），用于处理特征和名称

    提取此类以便QlibDataLoader和其他数据加载器（如QdbDataLoader）可以共享字段。
    """

    def __init__(self, config: Union[list, tuple, dict]):
        """
        参数
        ----------
        config : Union[list, tuple, dict]
            用于描述字段和列名的配置

            .. code-block::

                <config> := {
                    "group_name1": <fields_info1>
                    "group_name2": <fields_info2>
                }
                或
                <config> := <fields_info>

                <fields_info> := ["expr", ...] | (["expr", ...], ["col_name", ...])
                # 注意：列表或元组在解析时将被视为上述结构
        """
        self.is_group = isinstance(config, dict)

        if self.is_group:
            self.fields = {grp: self._parse_fields_info(fields_info) for grp, fields_info in config.items()}
        else:
            self.fields = self._parse_fields_info(config)

    def _parse_fields_info(self, fields_info: Union[list, tuple]) -> Tuple[list, list]:
        if len(fields_info) == 0:
            raise ValueError("字段大小必须大于0")

        if not isinstance(fields_info, (list, tuple)):
            raise TypeError("不支持的类型")

        if isinstance(fields_info[0], str):
            exprs = names = fields_info
        elif isinstance(fields_info[0], (list, tuple)):
            exprs, names = fields_info
        else:
            raise NotImplementedError(f"不支持这种输入类型")
        return exprs, names

    @abc.abstractmethod
    def load_group_df(
        self,
        instruments,
        exprs: list,
        names: list,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        gp_name: str = None,
    ) -> pd.DataFrame:
        """加载特定组的数据框

        参数
        ----------
        instruments :
            标的。
        exprs : list
            描述数据内容的表达式。
        names : list
            数据的名称。

        返回
        -------
        pd.DataFrame:
            查询到的数据框。
        """

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        if self.is_group:
            df = pd.concat(
                {
                    grp: self.load_group_df(instruments, exprs, names, start_time, end_time, grp)
                    for grp, (exprs, names) in self.fields.items()
                },
                axis=1,
            )
        else:
            exprs, names = self.fields
            df = self.load_group_df(instruments, exprs, names, start_time, end_time)
        return df


class QlibDataLoader(DLWParser):
    """
    与QlibDataLoader相同。可以通过配置定义字段。
    """

    def __init__(
        self,
        config: Tuple[list, tuple, dict],
        filter_pipe: List = None,
        swap_level: bool = True,
        freq: Union[str, dict] = "day",
        inst_processors: Union[dict, list] = None,
    ):
        """
        参数
        ----------
        config : Tuple[list, tuple, dict]
            请参考DLWParser的文档
        filter_pipe :
            标的过滤管道
        swap_level :
            是否交换多级索引的级别
        freq:  dict或str
            如果type(config) == dict且type(freq) == str，使用freq加载配置数据。
            如果type(config) == dict且type(freq) == dict，使用freq[<group_name>]加载config[<group_name>]数据
        inst_processors: dict | list
            如果inst_processors不为None且type(config) == dict；使用inst_processors[<group_name>]加载config[<group_name>]数据
            如果inst_processors是列表，则将应用于所有组。
        """
        self.filter_pipe = filter_pipe
        self.swap_level = swap_level
        self.freq = freq

        # sample
        self.inst_processors = inst_processors if inst_processors is not None else {}
        assert isinstance(
            self.inst_processors, (dict, list)
        ), f"inst_processors(={self.inst_processors}) must be dict or list"

        super().__init__(config)

        if self.is_group:
            # check sample config
            if isinstance(freq, dict):
                for _gp in config.keys():
                    if _gp not in freq:
                        raise ValueError(f"freq(={freq}) missing group(={_gp})")
                assert (
                    self.inst_processors
                ), f"freq(={self.freq}), inst_processors(={self.inst_processors}) cannot be None/empty"

    def load_group_df(
        self,
        instruments,
        exprs: list,
        names: list,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        gp_name: str = None,
    ) -> pd.DataFrame:
        if instruments is None:
            warnings.warn("`instruments` 未设置，将加载所有股票")
            instruments = "all"
        if isinstance(instruments, str):
            instruments = D.instruments(instruments, filter_pipe=self.filter_pipe)
        elif self.filter_pipe is not None:
            warnings.warn("`filter_pipe` 不为空，但当 `instruments` 为列表时不会使用它")

        freq = self.freq[gp_name] if isinstance(self.freq, dict) else self.freq
        inst_processors = (
            self.inst_processors if isinstance(self.inst_processors, list) else self.inst_processors.get(gp_name, [])
        )
        df = D.features(instruments, exprs, start_time, end_time, freq=freq, inst_processors=inst_processors)
        df.columns = names
        if self.swap_level:
            df = df.swaplevel().sort_index()  # 注意：如果交换级别，返回 <datetime, instrument>
        return df


class StaticDataLoader(DataLoader, Serializable):
    """支持从文件加载数据或直接提供数据的数据加载器。
    """

    include_attr = ["_config"]

    def __init__(self, config: Union[dict, str, pd.DataFrame], join="outer"):
        """参数
        ----------
        config : dict
            {字段组: <路径或对象>}
        join : str
            如何对齐不同的数据框
        """
        self._config = config  # using "_" to avoid confliction with the method `config` of Serializable
        self.join = join
        self._data = None

    def __getstate__(self) -> dict:
        # 避免序列化 `self._data`
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        self._maybe_load_raw_data()

        # 1) Filter by instruments
        if instruments is None:
            df = self._data
        else:
            df = self._data.loc(axis=0)[:, instruments]

        # 2) Filter by Datetime
        if start_time is None and end_time is None:
            return df  # 注意：避免通过loc复制
        # pd.Timestamp(None) == NaT，使用NaT作为索引无法获取正确的内容，所以不要改变None。
        start_time = time_to_slc_point(start_time)
        end_time = time_to_slc_point(end_time)
        return df.loc[start_time:end_time]

    def _maybe_load_raw_data(self):
        if self._data is not None:
            return
        if isinstance(self._config, dict):
            self._data = pd.concat(
                {fields_group: load_dataset(path_or_obj) for fields_group, path_or_obj in self._config.items()},
                axis=1,
                join=self.join,
            )
            self._data.sort_index(inplace=True)
        elif isinstance(self._config, (str, Path)):
            if str(self._config).strip().endswith(".parquet"):
                self._data = pd.read_parquet(self._config, engine="pyarrow")
            else:
                with Path(self._config).open("rb") as f:
                    self._data = pickle.load(f)
        elif isinstance(self._config, pd.DataFrame):
            self._data = self._config


class NestedDataLoader(DataLoader):
    """我们有多个数据加载器，可以使用此类来组合它们。
    """

    def __init__(self, dataloader_l: List[Dict], join="left") -> None:
        """参数
        ----------
        dataloader_l : list[dict]
            数据加载器列表，例如

            .. code-block:: python

                nd = NestedDataLoader(
                    dataloader_l=[
                        {
                            "class": "qlib.contrib.data.loader.Alpha158DL",
                        }, {
                            "class": "qlib.contrib.data.loader.Alpha360DL",
                            "kwargs": {
                                "config": {
                                    "label": ( ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
                                }
                            }
                        }
                    ]
                )
        join :
            在合并时将传递给 pd.concat。
        """
        super().__init__()
        self.data_loader_l = [
            (dl if isinstance(dl, DataLoader) else init_instance_by_config(dl)) for dl in dataloader_l
        ]
        self.join = join

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df_full = None
        for dl in self.data_loader_l:
            try:
                df_current = dl.load(instruments, start_time, end_time)
            except KeyError:
                warnings.warn(
                    "如果无法处理 `instruments` 的值，将把 instruments 设为 None 以获取所有数据。"
                )
                df_current = dl.load(instruments=None, start_time=start_time, end_time=end_time)
            if df_full is None:
                df_full = df_current
            else:
                current_columns = df_current.columns.tolist()
                full_columns = df_full.columns.tolist()
                columns_to_drop = [col for col in current_columns if col in full_columns]
                df_full.drop(columns=columns_to_drop, inplace=True)
                df_full = pd.merge(df_full, df_current, left_index=True, right_index=True, how=self.join)
        return df_full.sort_index(axis=1)


class DataLoaderDH(DataLoader):
    """DataLoaderDH
    基于数据处理器（Data Handler）的数据加载器
    它被设计用于从数据处理器加载多个数据
    - 如果你只想从单个数据处理器加载数据，可以在单个数据处理器中编写

    待办：是什么让这个模块不那么容易使用。

    - 对于在线场景

        - 底层数据处理器应该被配置。但数据加载器没有提供这样的接口和钩子。
    """

    def __init__(self, handler_config: dict, fetch_kwargs: dict = {}, is_group=False):
        """参数
        ----------
        handler_config : dict
            handler_config 将用于描述处理器

            .. code-block::

                <handler_config> := {
                    "group_name1": <handler>
                    "group_name2": <handler>
                }
                或
                <handler_config> := <handler>
                <handler> := 数据处理器实例 | 数据处理器配置

        fetch_kwargs : dict
            fetch_kwargs 将用于描述获取方法的不同参数，如 col_set、squeeze、data_key 等。

        is_group: bool
            is_group 将用于描述 handler_config 的键是否为组

        """
        from qlib.data.dataset.handler import DataHandler  # pylint: disable=C0415

        if is_group:
            self.handlers = {
                grp: init_instance_by_config(config, accept_types=DataHandler) for grp, config in handler_config.items()
            }
        else:
            self.handlers = init_instance_by_config(handler_config, accept_types=DataHandler)

        self.is_group = is_group
        self.fetch_kwargs = {"col_set": DataHandler.CS_RAW}
        self.fetch_kwargs.update(fetch_kwargs)

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        if instruments is not None:
            get_module_logger(self.__class__.__name__).warning(f"instruments[{instruments}] is ignored")

        if self.is_group:
            df = pd.concat(
                {
                    grp: dh.fetch(selector=slice(start_time, end_time), level="datetime", **self.fetch_kwargs)
                    for grp, dh in self.handlers.items()
                },
                axis=1,
            )
        else:
            df = self.handlers.fetch(selector=slice(start_time, end_time), level="datetime", **self.fetch_kwargs)
        return df
