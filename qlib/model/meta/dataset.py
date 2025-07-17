# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from qlib.model.meta.task import MetaTask
from typing import Dict, Union, List, Tuple, Text
from ...utils.serial import Serializable


class MetaTaskDataset(Serializable, metaclass=abc.ABCMeta):
    """
    元级别获取数据的数据集。

    元数据集负责:

    - 输入任务(如Qlib任务)并准备元任务

        - 元任务包含比普通任务更多的信息(如元模型的输入数据)

    学习到的模式可以迁移到其他元数据集。应支持以下情况:

    - 在元数据集A上训练的元模型然后应用于元数据集B

        - 元数据集A和B之间共享某些模式，因此在元数据集A上的元输入会在元模型应用于元数据集B时使用
    """

    def __init__(self, segments: Union[Dict[Text, Tuple], float], *args, **kwargs):
        """
        元数据集在初始化时维护一个元任务列表。

        segments参数指示数据划分方式

        MetaTaskDataset的`__init__`函数职责:
        - 初始化任务
        """
        super().__init__(*args, **kwargs)
        self.segments = segments

    def prepare_tasks(self, segments: Union[List[Text], Text], *args, **kwargs) -> List[MetaTask]:
        """
        准备每个元任务的数据以供训练。

        以下代码示例展示如何从`meta_dataset`获取元任务列表:

            .. code-block:: Python

                # 获取训练段和测试段，它们都是列表
                train_meta_tasks, test_meta_tasks = meta_dataset.prepare_tasks(["train", "test"])

        参数
        ----------
        segments: Union[List[Text], Tuple[Text], Text]
            选择数据的信息

        返回
        -------
        list:
            用于训练元模型的每个元任务准备数据的列表。对于多个段[seg1, seg2, ..., segN]，返回的列表将是[[seg1中的任务], [seg2中的任务], ..., [segN中的任务]]。
            每个任务都是一个元任务
        """
        if isinstance(segments, (list, tuple)):
            return [self._prepare_seg(seg) for seg in segments]
        elif isinstance(segments, str):
            return self._prepare_seg(segments)
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @abc.abstractmethod
    def _prepare_seg(self, segment: Text):
        """
        准备单个段的数据用于训练

        参数
        ----------
        seg : Text
            段的名称
        """
