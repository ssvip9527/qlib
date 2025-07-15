# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from typing import Dict, List, Text, Tuple, Union

import pandas as pd

from qlib.utils import init_instance_by_config

from ..data.dataset import Dataset
from ..data.dataset.utils import convert_index_format
from ..model.base import BaseModel
from ..utils.resam import resam_ts_data


class Signal(metaclass=abc.ABCMeta):
    """
    一些交易策略基于其他预测信号做出决策
    这些信号可能来自不同来源(例如准备好的数据、来自模型和数据集的在线预测)

    该接口试图为这些不同来源提供统一的接口
    """

    @abc.abstractmethod
    def get_signal(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Union[pd.Series, pd.DataFrame, None]:
        """
        获取决策步骤结束时的信号(从`start_time`到`end_time`)

        返回值
        -------
        Union[pd.Series, pd.DataFrame, None]:
            如果特定日期没有信号则返回None
        """


class SignalWCache(Signal):
    """
    带有基于pandas缓存的信号
    SignalWCache会将准备好的信号存储为属性，并根据输入查询提供相应的信号
    """

    def __init__(self, signal: Union[pd.Series, pd.DataFrame]) -> None:
        """
        参数
        ----------
        signal : Union[pd.Series, pd.DataFrame]
            信号的预期格式如下(索引顺序不重要，可以自动调整)

                instrument datetime
                SH600000   2008-01-02  0.079704
                           2008-01-03  0.120125
                           2008-01-04  0.878860
                           2008-01-07  0.505539
                           2008-01-08  0.395004
        """
        self.signal_cache = convert_index_format(signal, level="datetime")

    def get_signal(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Union[pd.Series, pd.DataFrame]:
        """
        获取指定时间范围内的信号数据
        
        参数:
            start_time: 开始时间
            end_time: 结束时间
            
        返回值:
            Union[pd.Series, pd.DataFrame]: 重采样后的信号数据
            
        说明:
            - 信号频率可能与策略决策频率不一致，因此需要进行数据重采样
            - 使用最新的信号数据(最近的数据)进行交易
        """
        signal = resam_ts_data(self.signal_cache, start_time=start_time, end_time=end_time, method="last")
        return signal


class ModelSignal(SignalWCache):
    def __init__(self, model: BaseModel, dataset: Dataset) -> None:
        self.model = model
        self.dataset = dataset
        pred_scores = self.model.predict(dataset)
        if isinstance(pred_scores, pd.DataFrame):
            pred_scores = pred_scores.iloc[:, 0]
        super().__init__(pred_scores)

    def _update_model(self) -> None:
        """
        使用在线数据时，按照以下步骤更新每个bar的模型：
            - 用在线数据更新数据集，数据集应支持在线更新
            - 生成新bar的最新预测分数
            - 将预测分数更新到最新预测中
        """
        # TODO: this method is not included in the framework and could be refactor later
        # 注意: 此方法尚未在框架中实现，后续可能重构
        raise NotImplementedError("_update_model is not implemented!")


def create_signal_from(
    obj: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame],
) -> Signal:
    """
    从多样化信息创建信号
    此方法会根据`obj`选择正确的方法创建信号
    请参考下面的代码。
    """
    if isinstance(obj, Signal):
        return obj
    elif isinstance(obj, (tuple, list)):
        return ModelSignal(*obj)
    elif isinstance(obj, (dict, str)):
        return init_instance_by_config(obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return SignalWCache(signal=obj)
    else:
        raise NotImplementedError(f"This type of signal is not supported")
