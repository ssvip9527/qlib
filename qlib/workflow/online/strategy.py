# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineStrategy模块是在线服务的一个组件。
"""

from typing import List, Union
from qlib.log import get_module_logger
from qlib.model.ens.group import RollingGroup
from qlib.utils import transform_end_date
from qlib.workflow.online.utils import OnlineTool, OnlineToolR
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.collect import Collector, RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import TimeAdjuster


class OnlineStrategy:
    """
    OnlineStrategy与`Online Manager <#Online Manager>`_配合使用，负责处理任务生成、模型更新和信号准备的方式。
    """

    def __init__(self, name_id: str):
        """
        初始化OnlineStrategy。
        此模块**必须**使用`Trainer <../reference/api.html#qlib.model.trainer.Trainer>`_来完成模型训练。

        参数:
            name_id (str): 唯一的名称或ID。
            trainer (qlib.model.trainer.Trainer, 可选): Trainer的实例。默认为None。
        """
        self.name_id = name_id
        self.logger = get_module_logger(self.__class__.__name__)
        self.tool = OnlineTool()

    def prepare_tasks(self, cur_time, **kwargs) -> List[dict]:
        """
        在例行程序结束后，根据当前时间（None表示最新）检查是否需要准备和训练一些新任务。
        返回等待训练的新任务。

        您可以通过OnlineTool.online_models找到最后的在线模型。
        """
        raise NotImplementedError(f"Please implement the `prepare_tasks` method.")

    def prepare_online_models(self, trained_models, cur_time=None) -> List[object]:
        """
        从训练好的模型中选择一些模型并将它们设置为在线模型。
        这是一个将所有训练好的模型设为在线的典型实现，您可以重写它来实现更复杂的方法。
        如果仍需要，可以通过OnlineTool.online_models找到最后的在线模型。

        注意：将所有在线模型重置为训练好的模型。如果没有训练好的模型，则不执行任何操作。

        **注意**:
            当前实现非常简单。以下是一个更接近实际场景的复杂情况：
            1. 在`test_start`前一天（时间戳`T`）训练新模型
            2. 在`test_start`时（通常是时间戳`T + 1`）切换模型

        参数:
            models (list): 模型列表。
            cur_time (pd.Dataframe): 来自OnlineManger的当前时间。None表示最新。

        返回:
            List[object]: 在线模型列表。
        """
        if not trained_models:
            return self.tool.online_models()
        self.tool.reset_online_tag(trained_models)
        return trained_models

    def first_tasks(self) -> List[dict]:
        """
        首先生成一系列任务并返回它们。
        """
        raise NotImplementedError(f"Please implement the `first_tasks` method.")

    def get_collector(self) -> Collector:
        """
        获取`Collector <../advanced/task_management.html#Task Collecting>`_实例以收集此策略的不同结果。

        例如:
            1) 在Recorder中收集预测
            2) 在txt文件中收集信号

        返回:
            Collector
        """
        raise NotImplementedError(f"Please implement the `get_collector` method.")


class RollingStrategy(OnlineStrategy):
    """
    此示例策略始终使用最新的滚动模型作为在线模型。
    """

    def __init__(
        self,
        name_id: str,
        task_template: Union[dict, List[dict]],
        rolling_gen: RollingGen,
    ):
        """
        初始化RollingStrategy。

        假设：name_id的字符串、实验名称和训练器的实验名称相同。

        参数:
            name_id (str): 唯一的名称或ID。也将作为实验的名称。
            task_template (Union[dict, List[dict]]): 任务模板列表或单个模板，将用于通过rolling_gen生成多个任务。
            rolling_gen (RollingGen): RollingGen的实例
        """
        super().__init__(name_id=name_id)
        self.exp_name = self.name_id
        if not isinstance(task_template, list):
            task_template = [task_template]
        self.task_template = task_template
        self.rg = rolling_gen
        assert issubclass(self.rg.__class__, RollingGen), "The rolling strategy relies on the feature if RollingGen"
        self.tool = OnlineToolR(self.exp_name)
        self.ta = TimeAdjuster()

    def get_collector(self, process_list=[RollingGroup()], rec_key_func=None, rec_filter_func=None, artifacts_key=None):
        """
        获取`Collector <../advanced/task_management.html#Task Collecting>`_实例以收集结果。返回的收集器必须能够区分不同模型的结果。

        假设：可以根据模型名称和滚动测试段来区分模型。
        如果不希望此假设，请实现您自己的方法或使用其他rec_key_func。

        参数:
            rec_key_func (Callable): 获取记录器键的函数。如果为None，则使用记录器ID。
            rec_filter_func (Callable, 可选): 通过返回True或False来过滤记录器。默认为None。
            artifacts_key (List[str], 可选): 要获取的工件键。如果为None，则获取所有工件。
        """

        def rec_key(recorder):
            task_config = recorder.load_object("task")
            model_key = task_config["model"]["class"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return model_key, rolling_key

        if rec_key_func is None:
            rec_key_func = rec_key

        artifacts_collector = RecorderCollector(
            experiment=self.exp_name,
            process_list=process_list,
            rec_key_func=rec_key_func,
            rec_filter_func=rec_filter_func,
            artifacts_key=artifacts_key,
        )

        return artifacts_collector

    def first_tasks(self) -> List[dict]:
        """
        Use rolling_gen to generate different tasks based on task_template.

        Returns:
            List[dict]: a list of tasks
        """
        return task_generator(
            tasks=self.task_template,
            generators=self.rg,  # generate different date segment
        )

    def prepare_tasks(self, cur_time) -> List[dict]:
        """
        Prepare new tasks based on cur_time (None for the latest).

        You can find the last online models by OnlineToolR.online_models.

        Returns:
            List[dict]: a list of new tasks.
        """
        # TODO: filter recorders by latest test segments is not a necessary
        latest_records, max_test = self._list_latest(self.tool.online_models())
        if max_test is None:
            self.logger.warn(f"No latest online recorders, no new tasks.")
            return []
        calendar_latest = transform_end_date(cur_time)
        self.logger.info(
            f"The interval between current time {calendar_latest} and last rolling test begin time {max_test[0]} is {self.ta.cal_interval(calendar_latest, max_test[0])}, the rolling step is {self.rg.step}"
        )
        res = []
        for rec in latest_records:
            task = rec.load_object("task")
            res.extend(self.rg.gen_following_tasks(task, calendar_latest))
        return res

    def _list_latest(self, rec_list: List[Recorder]):
        """
        List latest recorder form rec_list

        Args:
            rec_list (List[Recorder]): a list of Recorder

        Returns:
            List[Recorder], pd.Timestamp: the latest recorders and their test end time
        """
        if len(rec_list) == 0:
            return rec_list, None
        max_test = max(rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] for rec in rec_list)
        latest_rec = []
        for rec in rec_list:
            if rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] == max_test:
                latest_rec.append(rec)
        return latest_rec, max_test
