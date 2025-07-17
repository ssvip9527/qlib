# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
训练器(Trainer)用于训练一系列任务并返回模型记录器列表。
每个训练器包含两个步骤：
1. `train` - 创建模型记录器
2. `end_train` - 修改模型记录器

DelayTrainer是一种特殊训练器，可用于在线模拟并行训练：
- 第一步仅保存必要信息到记录器
- 第二步在最后执行并发耗时操作(如模型拟合)

Qlib提供两种训练器实现：
1. TrainerR - 基础训练器
2. TrainerRM - 基于TaskManager自动管理任务生命周期
"""

import socket
from typing import Callable, List, Optional

from tqdm.auto import tqdm

from qlib.config import C
from qlib.data.dataset import Dataset
from qlib.data.dataset.weight import Reweighter
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import (
    auto_filter_kwargs,
    fill_placeholder,
    flatten_dict,
    init_instance_by_config,
)
from qlib.utils.paral import call_in_subproc
from qlib.workflow import R
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.manage import TaskManager, run_task


def _log_task_info(task_config: dict):
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})  # keep the original format and datatype
    R.set_tags(**{"hostname": socket.gethostname()})


def _exe_task(task_config: dict):
    rec = R.get_recorder()
    # 模型和数据集初始化
    model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
    dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)
    reweighter: Reweighter = task_config.get("reweighter", None)
    # 模型训练
    auto_filter_kwargs(model.fit)(dataset, reweighter=reweighter)
    R.save_objects(**{"params.pkl": model})
    # 此数据集保存用于在线推理，因此不应转储具体数据
    dataset.config(dump_all=False, recursive=True)
    R.save_objects(**{"dataset": dataset})
    # 填充占位符
    placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
    task_config = fill_placeholder(task_config, placehorder_value)
    # 生成记录：预测、回测和分析
    records = task_config.get("record", [])
    if isinstance(records, dict):  # 防止只有一个字典
        records = [records]
    for record in records:
        # Some recorder require the parameter `model` and `dataset`.
        # try to automatically pass in them to the initialization function
        # to make defining the tasking easier
        r = init_instance_by_config(
            record,
            recorder=rec,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
        )
        r.generate()


def begin_task_train(task_config: dict, experiment_name: str, recorder_name: str = None) -> Recorder:
    """
    开始任务训练，创建记录器并保存任务配置。

    参数:
        task_config (dict): 任务配置
        experiment_name (str): 实验名称
        recorder_name (str): 记录器名称，None表示使用rid

    返回:
        Recorder: 模型记录器
    """
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        _log_task_info(task_config)
        return R.get_recorder()


def end_task_train(rec: Recorder, experiment_name: str) -> Recorder:
    """
    完成任务训练，执行实际的模型拟合和保存。

    参数:
        rec (Recorder): 需要恢复的记录器
        experiment_name (str): 实验名称

    返回:
        Recorder: 模型记录器
    """
    with R.start(experiment_name=experiment_name, recorder_id=rec.info["id"], resume=True):
        task_config = R.load_object("task")
        _exe_task(task_config)
    return rec


def task_train(task_config: dict, experiment_name: str, recorder_name: str = None) -> Recorder:
    """
    基于任务的训练，分为两个步骤执行

    参数
    ----------
    task_config : dict
        任务配置
    experiment_name: str
        实验名称
    recorder_name: str
        记录器名称

    返回
    ----------
    Recorder: 记录器实例
    """
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        _log_task_info(task_config)
        _exe_task(task_config)
        return R.get_recorder()


class Trainer:
    """
    训练器用于训练模型列表
    Trainer和DelayTrainer的区别在于完成实际训练的时机不同
    """

    def __init__(self):
        self.delay = False

    def train(self, tasks: list, *args, **kwargs) -> list:
        """
        给定任务定义列表，开始训练并返回模型。

        对于Trainer，此方法完成实际训练。
        对于DelayTrainer，此方法仅做准备工作。

        参数:
            tasks (list): 任务定义列表

        返回:
            list: 模型列表

        注意:
            - 对于`Trainer`，此方法将直接训练模型
            - 对于`DelayTrainer`，此方法仅做训练准备
        """
        raise NotImplementedError(f"Please implement the `train` method.")

    def end_train(self, models: list, *args, **kwargs) -> list:
        """
        给定模型列表，在训练结束时完成必要操作
        模型可能是记录器、文本文件、数据库等

        对于Trainer，该方法做一些收尾工作
        对于DelayTrainer，该方法完成实际训练

        参数:
            models: 模型列表

        返回:
            list: 模型列表
        """
        # do nothing if you finished all work in `train` method
        return models

    def is_delay(self) -> bool:
        """
        判断训练器是否会延迟完成`end_train`

        返回:
            bool: 是否为DelayTrainer
        """
        return self.delay

    def __call__(self, *args, **kwargs) -> list:
        return self.end_train(self.train(*args, **kwargs))

    def has_worker(self) -> bool:
        """
        判断是否启用了并行训练的后台工作器

        返回
        -------
        bool:
            工作器是否启用
        """
        return False

    def worker(self):
        """
        启动工作进程

        异常:
            NotImplementedError: 如果不支持工作进程
        """
        raise NotImplementedError(f"Please implement the `worker` method")


class TrainerR(Trainer):
    """
    基于记录器(R)的训练器
    以线性方式训练任务列表并返回模型记录器列表

    假设：模型由`task`定义，结果将保存到`Recorder`
    """

    # Those tag will help you distinguish whether the Recorder has finished traning
    # 这些标签将帮助你区分 Recorder 是否已完成训练
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        train_func: Callable = task_train,
        call_in_subproc: bool = False,
        default_rec_name: Optional[str] = None,
    ):
        """
        初始化TrainerR

        参数:
            experiment_name (str, optional): 默认实验名称
            train_func (Callable, optional): 默认训练方法，默认为`task_train`
            call_in_subproc (bool): 在子进程中调用以强制释放内存
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.default_rec_name = default_rec_name
        self.train_func = train_func
        self._call_in_subproc = call_in_subproc

    def train(
        self, tasks: list, train_func: Optional[Callable] = None, experiment_name: Optional[str] = None, **kwargs
    ) -> List[Recorder]:
        """
        给定任务列表并返回训练好的记录器列表，顺序可以保证。

        参数:
            tasks (list): 基于任务字典的定义列表
            train_func (Callable): 训练方法，至少需要tasks和experiment_name参数。None表示使用默认训练方法。
            experiment_name (str): 实验名称，None表示使用默认名称。
            kwargs: train_func的参数。

        返回:
            List[Recorder]: 记录器列表
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        recs = []
        for task in tqdm(tasks, desc="train tasks"):
            if self._call_in_subproc:
                get_module_logger("TrainerR").info("running models in sub process (for forcing release memroy).")
                train_func = call_in_subproc(train_func, C)
            rec = train_func(task, experiment_name, recorder_name=self.default_rec_name, **kwargs)
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            recs.append(rec)
        return recs

    def end_train(self, models: list, **kwargs) -> List[Recorder]:
        """
        为记录器设置STATUS_END标签

        参数:
            models (list): 训练好的记录器列表

        返回:
            List[Recorder]: 与参数相同的列表
        """
        if isinstance(models, Recorder):
            models = [models]
        for rec in models:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return models


class DelayTrainerR(TrainerR):
    """
    基于TrainerR的延迟实现，意味着`train`方法可能只做准备工作，而`end_train`方法完成实际的模型拟合
    """

    def __init__(
        self, experiment_name: str = None, train_func=begin_task_train, end_train_func=end_task_train, **kwargs
    ):
        """
        初始化TrainerRM

        参数:
            experiment_name (str): 默认实验名称
            train_func (Callable, optional): 默认训练方法，默认为`begin_task_train`
            end_train_func (Callable, optional): 默认结束训练方法，默认为`end_task_train`
        """
        super().__init__(experiment_name, train_func, **kwargs)
        self.end_train_func = end_train_func
        self.delay = True

    def end_train(self, models, end_train_func=None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        给定记录器列表并返回训练好的记录器列表
        该类将完成实际的数据加载和模型拟合

        参数:
            models (list): 记录器列表，任务已保存到其中
            end_train_func (Callable, optional): 结束训练方法，至少需要`recorders`和`experiment_name`参数，默认为None表示使用self.end_train_func
            experiment_name (str): 实验名称，None表示使用默认名称
            kwargs: end_train_func的参数

        返回:
            List[Recorder]: 记录器列表
        """
        if isinstance(models, Recorder):
            models = [models]
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        for rec in models:
            if rec.list_tags()[self.STATUS_KEY] == self.STATUS_END:
                continue
            end_train_func(rec, experiment_name, **kwargs)
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return models


class TrainerRM(Trainer):
    """
    基于记录器(R)和任务管理器(M)的训练器
    可以以多进程方式训练任务列表并返回模型记录器列表

    假设：`task`将保存到TaskManager，并且`task`将从TaskManager获取并训练
    """

    # Those tag will help you distinguish whether the Recorder has finished traning
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"

    # This tag is the _id in TaskManager to distinguish tasks.
    TM_ID = "_id in TaskManager"

    def __init__(
        self,
        experiment_name: str = None,
        task_pool: str = None,
        train_func=task_train,
        skip_run_task: bool = False,
        default_rec_name: Optional[str] = None,
    ):
        """
        初始化TrainerR

        参数:
            experiment_name (str): 默认实验名称
            task_pool (str): TaskManager中的任务池名称，None表示使用与experiment_name相同的名称
            train_func (Callable, optional): 默认训练方法，默认为`task_train`
            skip_run_task (bool):
                如果skip_run_task == True:
                仅在worker中运行run_task，否则跳过run_task
        """

        super().__init__()
        self.experiment_name = experiment_name
        self.task_pool = task_pool
        self.train_func = train_func
        self.skip_run_task = skip_run_task
        self.default_rec_name = default_rec_name

    def train(
        self,
        tasks: list,
        train_func: Callable = None,
        experiment_name: str = None,
        before_status: str = TaskManager.STATUS_WAITING,
        after_status: str = TaskManager.STATUS_DONE,
        default_rec_name: Optional[str] = None,
        **kwargs,
    ) -> List[Recorder]:
        """
        给定任务列表并返回训练好的记录器列表，顺序可以保证。

        此方法默认为单进程，但TaskManager提供了并行训练的强大方式。
        用户可以自定义train_func实现多进程甚至多机器训练。

        参数:
            tasks (list): 基于任务字典的定义列表
            train_func (Callable): 训练方法，至少需要tasks和experiment_name参数。None表示使用默认训练方法。
            experiment_name (str): 实验名称，None表示使用默认名称。
            before_status (str): 处于before_status状态的任务将被获取并训练。可以是STATUS_WAITING, STATUS_PART_DONE。
            after_status (str): 训练后的任务将变为after_status状态。可以是STATUS_WAITING, STATUS_PART_DONE。
            kwargs: train_func的参数。

        返回:
            List[Recorder]: 记录器列表
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        if default_rec_name is None:
            default_rec_name = self.default_rec_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        tm = TaskManager(task_pool=task_pool)
        _id_list = tm.create_task(tasks)  # all tasks will be saved to MongoDB
        query = {"_id": {"$in": _id_list}}
        if not self.skip_run_task:
            run_task(
                train_func,
                task_pool,
                query=query,  # only train these tasks
                experiment_name=experiment_name,
                before_status=before_status,
                after_status=after_status,
                recorder_name=default_rec_name,
                **kwargs,
            )

        if not self.is_delay():
            tm.wait(query=query)

        recs = []
        for _id in _id_list:
            rec = tm.re_query(_id)["res"]
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            rec.set_tags(**{self.TM_ID: _id})
            recs.append(rec)
        return recs

    def end_train(self, recs: list, **kwargs) -> List[Recorder]:
        """
        为记录器设置STATUS_END标签。

        参数:
            recs (list): 训练好的记录器列表。

        返回:
            List[Recorder]: 与参数相同的列表。
        """
        if isinstance(recs, Recorder):
            recs = [recs]
        for rec in recs:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return recs

    def worker(
        self,
        train_func: Callable = None,
        experiment_name: str = None,
    ):
        """
        `train`方法的多进程实现。可以与`train`共享同一个task_pool，并能在其他进程或其他机器上运行。

        参数:
            train_func (Callable): 训练方法，至少需要tasks和experiment_name参数。None表示使用默认训练方法。
            experiment_name (str): 实验名称，None表示使用默认名称。
        """
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        run_task(train_func, task_pool=task_pool, experiment_name=experiment_name)

    def has_worker(self) -> bool:
        return True


class DelayTrainerRM(TrainerRM):
    """
    基于TrainerRM的延迟实现，意味着`train`方法可能只做准备工作，而`end_train`方法完成实际模型拟合。

    """

    def __init__(
        self,
        experiment_name: str = None,
        task_pool: str = None,
        train_func=begin_task_train,
        end_train_func=end_task_train,
        skip_run_task: bool = False,
        **kwargs,
    ):
        """
        初始化DelayTrainerRM。

        参数:
            experiment_name (str): 默认实验名称。
            task_pool (str): TaskManager中的任务池名称。None表示使用与experiment_name相同的名称。
            train_func (Callable, optional): 默认训练方法。默认为`begin_task_train`。
            end_train_func (Callable, optional): 默认结束训练方法。默认为`end_task_train`。
            skip_run_task (bool):
                如果skip_run_task == True:
                仅在worker中运行run_task。否则跳过run_task。
                例如：在CPU虚拟机上启动训练器，然后等待任务在GPU虚拟机上完成。
        """
        super().__init__(experiment_name, task_pool, train_func, **kwargs)
        self.end_train_func = end_train_func
        self.delay = True
        self.skip_run_task = skip_run_task

    def train(self, tasks: list, train_func=None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        与TrainerRM的`train`方法相同，after_status将为STATUS_PART_DONE。

        参数:
            tasks (list): 基于任务字典的定义列表
            train_func (Callable): 训练方法，至少需要tasks和experiment_name参数。None表示使用self.train_func。
            experiment_name (str): 实验名称，None表示使用默认名称。

        返回:
            List[Recorder]: 记录器列表
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        _skip_run_task = self.skip_run_task
        self.skip_run_task = False  # The task preparation can't be skipped
        res = super().train(
            tasks,
            train_func=train_func,
            experiment_name=experiment_name,
            after_status=TaskManager.STATUS_PART_DONE,
            **kwargs,
        )
        self.skip_run_task = _skip_run_task
        return res

    def end_train(self, recs, end_train_func=None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        给定记录器列表并返回训练好的记录器列表。
        此类将完成实际数据加载和模型拟合。

        参数:
            recs (list): 记录器列表，任务已保存到其中。
            end_train_func (Callable, optional): 结束训练方法，至少需要recorders和experiment_name参数。None表示使用self.end_train_func。
            experiment_name (str): 实验名称，None表示使用默认名称。
            kwargs: end_train_func的参数。

        返回:
            List[Recorder]: 记录器列表
        """
        if isinstance(recs, Recorder):
            recs = [recs]
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        _id_list = []
        for rec in recs:
            _id_list.append(rec.list_tags()[self.TM_ID])

        query = {"_id": {"$in": _id_list}}
        if not self.skip_run_task:
            run_task(
                end_train_func,
                task_pool,
                query=query,  # only train these tasks
                experiment_name=experiment_name,
                before_status=TaskManager.STATUS_PART_DONE,
                **kwargs,
            )

        TaskManager(task_pool=task_pool).wait(query=query)

        for rec in recs:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return recs

    def worker(self, end_train_func=None, experiment_name: str = None):
        """
        `end_train`方法的多进程实现。可以与`end_train`共享同一个task_pool，并能在其他进程或其他机器上运行。

        参数:
            end_train_func (Callable, optional): 结束训练方法，至少需要recorders和experiment_name参数。None表示使用self.end_train_func。
            experiment_name (str): 实验名称，None表示使用默认名称。
        """
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        run_task(
            end_train_func,
            task_pool=task_pool,
            experiment_name=experiment_name,
            before_status=TaskManager.STATUS_PART_DONE,
        )

    def has_worker(self) -> bool:
        return True
