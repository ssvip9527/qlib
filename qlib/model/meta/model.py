# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import List

from .dataset import MetaTaskDataset


class MetaModel(metaclass=abc.ABCMeta):
    """
    指导模型学习的元模型。

    `Guiding`一词根据模型学习阶段可分为两种类型:
    - 学习任务的定义: 请参考`MetaTaskModel`的文档
    - 控制模型的学习过程: 请参考`MetaGuideModel`的文档
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        元模型的训练过程。
        """

    @abc.abstractmethod
    def inference(self, *args, **kwargs) -> object:
        """
        元模型的推理过程。

        返回
        -------
        object:
            用于指导模型学习的一些信息
        """


class MetaTaskModel(MetaModel):
    """
    这类元模型处理基础任务定义。元模型在训练后会创建用于训练新基础预测模型的任务。`prepare_tasks`直接修改任务定义。
    """

    def fit(self, meta_dataset: MetaTaskDataset):
        """
        MetaTaskModel预期从meta_dataset获取准备好的MetaTask。
        然后它将从元任务中学习知识
        """
        raise NotImplementedError(f"Please implement the `fit` method")

    def inference(self, meta_dataset: MetaTaskDataset) -> List[dict]:
        """
        MetaTaskModel将对meta_dataset进行推理
        MetaTaskModel预期从meta_dataset获取准备好的MetaTask。
        然后它将创建可被Qlib训练器执行的Qlib格式的修改后任务。

        返回
        -------
        List[dict]:
            修改后的任务定义列表。

        """
        raise NotImplementedError(f"Please implement the `inference` method")


class MetaGuideModel(MetaModel):
    """
    这类元模型旨在指导基础模型的训练过程。元模型在基础预测模型训练过程中与其交互。
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs):
        pass
