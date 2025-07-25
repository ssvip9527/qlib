# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
TaskGenerator模块可以根据TaskGen和一些任务模板生成多个任务
"""
import abc
import copy
import pandas as pd
from typing import Dict, List, Union, Callable

from qlib.utils import transform_end_date
from .utils import TimeAdjuster


def task_generator(tasks, generators) -> list:
    """
    使用TaskGen列表和任务模板列表生成不同的任务

    例如:

        有3个任务模板a,b,c和2个TaskGen A,B。A会从每个模板生成2个任务，B会生成3个任务
        task_generator([a, b, c], [A, B])最终会生成3*2*3 = 18个任务

    参数
    ----------
    tasks : List[dict] or dict
        任务模板列表或单个任务
    generators : List[TaskGen] or TaskGen
        TaskGen列表或单个TaskGen

    返回
    -------
    list
        任务列表
    """

    if isinstance(tasks, dict):
        tasks = [tasks]
    if isinstance(generators, TaskGen):
        generators = [generators]

    # generate gen_task_list
    for gen in generators:
        new_task_list = []
        for task in tasks:
            new_task_list.extend(gen.generate(task))
        tasks = new_task_list

    return tasks


class TaskGen(metaclass=abc.ABCMeta):
    """
    生成不同任务的基类

    示例1:

        输入: 一个特定的任务模板和滚动步骤

        输出: 任务的滚动版本

    示例2:

        输入: 一个特定的任务模板和损失列表

        输出: 一组具有不同损失的任务

    """

    @abc.abstractmethod
    def generate(self, task: dict) -> List[dict]:
        """
        Generate different tasks based on a task template

        Parameters
        ----------
        task: dict
            a task template

        Returns
        -------
        typing.List[dict]:
            A list of tasks
        """

    def __call__(self, *args, **kwargs):
        """
        This is just a syntactic sugar for generate
        """
        return self.generate(*args, **kwargs)


def handler_mod(task: dict, rolling_gen):
    """
    在使用RollingGen时帮助修改handler的结束时间
    尝试处理以下情况:

    - Handler的数据end_time早于dataset的test_data的segments

        - 为此需要扩展handler数据的end_time

    如果handler的end_time为None，则无需更改其结束时间

    参数:
        task (dict): 任务模板
        rg (RollingGen): RollingGen实例
    """
    try:
        interval = rolling_gen.ta.cal_interval(
            task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"],
            task["dataset"]["kwargs"]["segments"][rolling_gen.test_key][1],
        )
        # if end_time < the end of test_segments, then change end_time to allow load more data
        if interval < 0:
            task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = copy.deepcopy(
                task["dataset"]["kwargs"]["segments"][rolling_gen.test_key][1]
            )
    except KeyError:
        # Maybe dataset do not have handler, then do nothing.
        pass
    except TypeError:
        # May be the handler is a string. `"handler.pkl"["kwargs"]` will raise TypeError
        # e.g. a dumped file like file:///<file>/
        pass


def trunc_segments(ta: TimeAdjuster, segments: Dict[str, pd.Timestamp], days, test_key="test"):
    """
    为避免未来信息泄露，应根据测试开始时间截断segments

    注意:
        此函数将**原地**修改segments
    """
    # adjust segment
    test_start = min(t for t in segments[test_key] if t is not None)
    for k in list(segments.keys()):
        if k != test_key:
            segments[k] = ta.truncate(segments[k], test_start, days)


class RollingGen(TaskGen):
    ROLL_EX = TimeAdjuster.SHIFT_EX  # fixed start date, expanding end date
    ROLL_SD = TimeAdjuster.SHIFT_SD  # fixed segments size, slide it from start date

    def __init__(
        self,
        step: int = 40,
        rtype: str = ROLL_EX,
        ds_extra_mod_func: Union[None, Callable] = handler_mod,
        test_key="test",
        train_key="train",
        trunc_days: int = None,
        task_copy_func: Callable = copy.deepcopy,
    ):
        """
        Generate tasks for rolling

        Parameters
        ----------
        step : int
            step to rolling
        rtype : str
            rolling type (expanding, sliding)
        ds_extra_mod_func: Callable
            A method like: handler_mod(task: dict, rg: RollingGen)
            Do some extra action after generating a task. For example, use ``handler_mod`` to modify the end time of the handler of a dataset.
        trunc_days: int
            trunc some data to avoid future information leakage
        task_copy_func: Callable
            the function to copy entire task. This is very useful when user want to share something between tasks
        """
        self.step = step
        self.rtype = rtype
        self.ds_extra_mod_func = ds_extra_mod_func
        self.ta = TimeAdjuster(future=True)

        self.test_key = test_key
        self.train_key = train_key
        self.trunc_days = trunc_days
        self.task_copy_func = task_copy_func

    def _update_task_segs(self, task, segs):
        # update segments of this task
        task["dataset"]["kwargs"]["segments"] = copy.deepcopy(segs)
        if self.ds_extra_mod_func is not None:
            self.ds_extra_mod_func(task, self)

    def gen_following_tasks(self, task: dict, test_end: pd.Timestamp) -> List[dict]:
        """
        generating following rolling tasks for `task` until test_end

        Parameters
        ----------
        task : dict
            Qlib task format
        test_end : pd.Timestamp
            the latest rolling task includes `test_end`

        Returns
        -------
        List[dict]:
            the following tasks of `task`(`task` itself is excluded)
        """
        prev_seg = task["dataset"]["kwargs"]["segments"]
        while True:
            segments = {}
            try:
                for k, seg in prev_seg.items():
                    # decide how to shift
                    # expanding only for train data, the segments size of test data and valid data won't change
                    if k == self.train_key and self.rtype == self.ROLL_EX:
                        rtype = self.ta.SHIFT_EX
                    else:
                        rtype = self.ta.SHIFT_SD
                    # shift the segments data
                    segments[k] = self.ta.shift(seg, step=self.step, rtype=rtype)
                if segments[self.test_key][0] > test_end:
                    break
            except KeyError:
                # We reach the end of tasks
                # No more rolling
                break

            prev_seg = segments
            t = self.task_copy_func(task)  # deepcopy is necessary to avoid replace task inplace
            self._update_task_segs(t, segments)
            yield t

    def generate(self, task: dict) -> List[dict]:
        """
        Converting the task into a rolling task.

        Parameters
        ----------
        task: dict
            A dict describing a task. For example.

            .. code-block:: python

                DEFAULT_TASK = {
                    "model": {
                        "class": "LGBModel",
                        "module_path": "qlib.contrib.model.gbdt",
                    },
                    "dataset": {
                        "class": "DatasetH",
                        "module_path": "qlib.data.dataset",
                        "kwargs": {
                            "handler": {
                                "class": "Alpha158",
                                "module_path": "qlib.contrib.data.handler",
                                "kwargs": {
                                    "start_time": "2008-01-01",
                                    "end_time": "2020-08-01",
                                    "fit_start_time": "2008-01-01",
                                    "fit_end_time": "2014-12-31",
                                    "instruments": "csi100",
                                },
                            },
                            "segments": {
                                "train": ("2008-01-01", "2014-12-31"),
                                "valid": ("2015-01-01", "2016-12-20"),  # Please avoid leaking the future test data into validation
                                "test": ("2017-01-01", "2020-08-01"),
                            },
                        },
                    },
                    "record": [
                        {
                            "class": "SignalRecord",
                            "module_path": "qlib.workflow.record_temp",
                        },
                    ]
                }

        Returns
        ----------
        List[dict]: a list of tasks
        """
        res = []

        t = self.task_copy_func(task)

        # calculate segments

        # First rolling
        # 1) prepare the end point
        segments: dict = copy.deepcopy(self.ta.align_seg(t["dataset"]["kwargs"]["segments"]))
        test_end = transform_end_date(segments[self.test_key][1])
        # 2) and init test segments
        test_start_idx = self.ta.align_idx(segments[self.test_key][0])
        segments[self.test_key] = (self.ta.get(test_start_idx), self.ta.get(test_start_idx + self.step - 1))
        if self.trunc_days is not None:
            trunc_segments(self.ta, segments, self.trunc_days, self.test_key)

        # update segments of this task
        self._update_task_segs(t, segments)

        res.append(t)

        # Update the following rolling
        res.extend(self.gen_following_tasks(t, test_end))
        return res


class MultiHorizonGenBase(TaskGen):
    def __init__(self, horizon: List[int] = [5], label_leak_n=2):
        """
        基于现有任务为不同周期生成任务

        参数
        ----------
        horizon : List[int]
            任务可能的周期列表
        label_leak_n : int
            预测日后需要多少天才能获得完整标签
            例如:
            - 用户在`T`日(收盘后)进行预测
            - 标签是在`T + 1`日买入并在`T + 2`日卖出的收益
            - `label_leak_n`将为2(即需要泄露2天的信息来利用该样本)
        """
        self.horizon = list(horizon)
        self.label_leak_n = label_leak_n
        self.ta = TimeAdjuster()
        self.test_key = "test"

    @abc.abstractmethod
    def set_horizon(self, task: dict, hr: int):
        """
        此方法用于**原地**修改任务

        参数
        ----------
        task : dict
            Qlib任务
        hr : int
            任务周期
        """

    def generate(self, task: dict):
        res = []
        for hr in self.horizon:
            # Add horizon
            t = copy.deepcopy(task)
            self.set_horizon(t, hr)

            # adjust segment
            segments = self.ta.align_seg(t["dataset"]["kwargs"]["segments"])
            trunc_segments(self.ta, segments, days=hr + self.label_leak_n, test_key=self.test_key)
            t["dataset"]["kwargs"]["segments"] = segments
            res.append(t)
        return res
