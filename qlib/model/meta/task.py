# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.data.dataset import Dataset
from ...utils import init_instance_by_config


class MetaTask:
    """
    单个元任务，元数据集包含一个元任务列表。
    它作为MetaDatasetDS中的一个组件。

    数据处理方式不同:

    - 训练和测试时的处理输入可能不同

        - 训练时，训练任务中的X、y、X_test、y_test是必要的(# PROC_MODE_FULL #)
          但在测试任务中不是必须的。(# PROC_MODE_TEST #)
        - 当元模型可以迁移到其他数据集时，只需要meta_info (# PROC_MODE_TRANSFER #)
    """

    PROC_MODE_FULL = "full"
    PROC_MODE_TEST = "test"
    PROC_MODE_TRANSFER = "transfer"

    def __init__(self, task: dict, meta_info: object, mode: str = PROC_MODE_FULL):
        """
        `__init__`函数负责:

        - 存储任务
        - 存储原始输入数据
        - 处理元数据的输入数据

        参数
        ----------
        task : dict
            需要被元模型增强的任务

        meta_info : object
            元模型的输入
        """
        self.task = task
        self.meta_info = meta_info  # the original meta input information, it will be processed later
        self.mode = mode

    def get_dataset(self) -> Dataset:
        return init_instance_by_config(self.task["dataset"], accept_types=Dataset)

    def get_meta_input(self) -> object:
        """
        返回**处理后的**meta_info
        """
        return self.meta_info

    def __repr__(self):
        return f"MetaTask(task={self.task}, meta_info={self.meta_info})"
