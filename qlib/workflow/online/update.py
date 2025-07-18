# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
更新器模块，用于在股票数据更新时更新预测等artifact。
"""

from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd
from qlib import get_module_logger
from qlib.data import D
from qlib.data.dataset import Dataset, DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model import Model
from qlib.utils import get_date_by_shift
from qlib.workflow.recorder import Recorder
from qlib.workflow.record_temp import SignalRecord


class RMDLoader:
    """
    Recorder Model Dataset Loader
    """

    def __init__(self, rec: Recorder):
        self.rec = rec

    def get_dataset(
        self, start_time, end_time, segments=None, unprepared_dataset: Optional[DatasetH] = None
    ) -> DatasetH:
        """
        加载、配置和设置数据集

        该数据集用于推理

        参数:
            start_time :
                基础数据的开始时间
            end_time :
                基础数据的结束时间
            segments : dict
                数据集的分段配置
                对于时间序列数据集(TSDatasetH)，测试段可能与开始时间和结束时间不同
            unprepared_dataset: Optional[DatasetH]
                如果用户不想从记录器加载数据集，请指定用户的数据集

        返回:
            DatasetH: DatasetH实例

        """
        if segments is None:
            segments = {"test": (start_time, end_time)}
        if unprepared_dataset is None:
            dataset: DatasetH = self.rec.load_object("dataset")
        else:
            dataset = unprepared_dataset
        dataset.config(handler_kwargs={"start_time": start_time, "end_time": end_time}, segments=segments)
        dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
        return dataset

    def get_model(self) -> Model:
        return self.rec.load_object("params.pkl")


class RecordUpdater(metaclass=ABCMeta):
    """
    Update a specific recorders
    """

    def __init__(self, record: Recorder, *args, **kwargs):
        self.record = record
        self.logger = get_module_logger(self.__class__.__name__)

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update info for specific recorder
        """


class DSBasedUpdater(RecordUpdater, metaclass=ABCMeta):
    """
    基于数据集的更新器

    - 提供基于Qlib数据集更新数据的功能

    假设条件

    - 基于Qlib数据集
    - 要更新的数据是多级索引的pd.DataFrame，例如标签、预测

        .. code-block::

                                     LABEL0
            datetime   instrument
            2021-05-10 SH600000    0.006965
                       SH600004    0.003407
            ...                         ...
            2021-05-28 SZ300498    0.015748
                       SZ300676   -0.001321
    """

    def __init__(
        self,
        record: Recorder,
        to_date=None,
        from_date=None,
        hist_ref: Optional[int] = None,
        freq="day",
        fname="pred.pkl",
        loader_cls: type = RMDLoader,
    ):
        """
        初始化预测更新器

        在以下情况下的预期行为:

        - 如果`to_date`大于日历中的最大日期，数据将更新到最新日期
        - 如果有数据在`from_date`之前或`to_date`之后，只有`from_date`和`to_date`之间的数据会受到影响

        参数:
            record : Recorder
                记录器
            to_date :
                更新预测到`to_date`

                如果to_date为None:

                    数据将更新到最新日期
            from_date :
                更新将从`from_date`开始

                如果from_date为None:

                    更新将在历史数据中最新数据的下一个时间点进行
            hist_ref : int
                有时数据集会有历史依赖
                将历史依赖长度的问题留给用户设置
                如果用户不指定此参数，更新器将尝试加载数据集自动确定hist_ref

                .. note::

                    start_time不包含在`hist_ref`中；因此`hist_ref`在大多数情况下会是`step_len - 1`

            loader_cls : type
                加载模型和数据集的类

        """
        # TODO: automate this hist_ref in the future.
        super().__init__(record=record)

        self.to_date = to_date
        self.hist_ref = hist_ref
        self.freq = freq
        self.fname = fname
        self.rmdl = loader_cls(rec=record)

        latest_date = D.calendar(freq=freq)[-1]
        if to_date is None:
            to_date = latest_date
        to_date = pd.Timestamp(to_date)

        if to_date >= latest_date:
            self.logger.warning(
                f"The given `to_date`({to_date}) is later than `latest_date`({latest_date}). So `to_date` is clipped to `latest_date`."
            )
            to_date = latest_date
        self.to_date = to_date

        # FIXME: it will raise error when running routine with delay trainer
        # should we use another prediction updater for delay trainer?
        self.old_data: pd.DataFrame = record.load_object(fname)
        if from_date is None:
            # dropna is for being compatible to some data with future information(e.g. label)
            # The recent label data should be updated together
            self.last_end = self.old_data.dropna().index.get_level_values("datetime").max()
        else:
            self.last_end = get_date_by_shift(from_date, -1, align="right")

    def prepare_data(self, unprepared_dataset: Optional[DatasetH] = None) -> DatasetH:
        """
        加载数据集
        - 如果指定了unprepared_dataset，则直接准备数据集
        - 否则

        分离此函数将使重用数据集更容易

        返回:
            DatasetH: DatasetH实例
        """
        # automatically getting the historical dependency if not specified
        if self.hist_ref is None:
            dataset: DatasetH = self.record.load_object("dataset") if unprepared_dataset is None else unprepared_dataset
            # Special treatment of historical dependencies
            if isinstance(dataset, TSDatasetH):
                hist_ref = dataset.step_len - 1
            else:
                hist_ref = 0  # if only the lastest data is used, then only current data will be used and no historical data will be used
        else:
            hist_ref = self.hist_ref

        start_time_buffer = get_date_by_shift(
            self.last_end, -hist_ref + 1, clip_shift=False, freq=self.freq  # pylint: disable=E1130
        )
        start_time = get_date_by_shift(self.last_end, 1, freq=self.freq)
        seg = {"test": (start_time, self.to_date)}
        return self.rmdl.get_dataset(
            start_time=start_time_buffer, end_time=self.to_date, segments=seg, unprepared_dataset=unprepared_dataset
        )

    def update(self, dataset: DatasetH = None, write: bool = True, ret_new: bool = False) -> Optional[object]:
        """
        参数
        ----------
        dataset : DatasetH
            DatasetH实例。None表示需要重新准备
        write : bool
            是否执行写入操作
        ret_new : bool
            是否返回更新后的数据

        返回
        -------
        Optional[object]
            更新后的数据集
        """
        # FIXME: the problem below is not solved
        # The model dumped on GPU instances can not be loaded on CPU instance. Follow exception will raised
        # RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        # https://github.com/pytorch/pytorch/issues/16797

        if self.last_end >= self.to_date:
            self.logger.info(
                f"The data in {self.record.info['id']} are latest ({self.last_end}). No need to update to {self.to_date}."
            )
            return

        # load dataset
        if dataset is None:
            # For reusing the dataset
            dataset = self.prepare_data()

        updated_data = self.get_update_data(dataset)

        if write:
            self.record.save_objects(**{self.fname: updated_data})
        if ret_new:
            return updated_data

    @abstractmethod
    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        """
        基于给定数据集返回更新后的数据

        `get_update_data`和`update`的区别
        - `update_date`只包含一些数据特定的功能
        - `update`包含一些常规步骤(例如准备数据集、检查)
        """


def _replace_range(data, new_data):
    dates = new_data.index.get_level_values("datetime")
    data = data.sort_index()
    data = data.drop(data.loc[dates.min() : dates.max()].index)
    cb_data = pd.concat([data, new_data], axis=0)
    cb_data = cb_data[~cb_data.index.duplicated(keep="last")].sort_index()
    return cb_data


class PredUpdater(DSBasedUpdater):
    """
    更新记录器中的预测
    """

    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        # Load model
        model = self.rmdl.get_model()
        new_pred: pd.Series = model.predict(dataset)
        data = _replace_range(self.old_data, new_pred.to_frame("score"))
        self.logger.info(f"Finish updating new {new_pred.shape[0]} predictions in {self.record.info['id']}.")
        return data


class LabelUpdater(DSBasedUpdater):
    """
    更新记录器中的标签

    假设条件
    - 标签由record_temp.SignalRecord生成
    """

    def __init__(self, record: Recorder, to_date=None, **kwargs):
        super().__init__(record, to_date=to_date, fname="label.pkl", **kwargs)

    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        new_label = SignalRecord.generate_label(dataset)
        cb_data = _replace_range(self.old_data.sort_index(), new_label)
        return cb_data
