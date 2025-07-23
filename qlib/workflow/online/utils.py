# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineTool是一个用于设置和取消设置一系列`online`模型的模块。
`online`模型是在某些时间点的决定性模型，可以随时间变化而改变。
这使我们能够使用高效的子模型来适应市场风格的变化。
"""

from typing import List, Union

from qlib.log import get_module_logger
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow.online.update import PredUpdater
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.utils import list_recorders


class OnlineTool:
    """
    OnlineTool将管理包含模型记录器的实验中的`online`模型。
    """

    ONLINE_KEY = "online_status"  # 记录器中的在线状态键
    ONLINE_TAG = "online"  # '在线'模型
    OFFLINE_TAG = "offline"  # '离线'模型，不用于在线服务

    def __init__(self):
        """
        初始化OnlineTool。
        """
        self.logger = get_module_logger(self.__class__.__name__)

    def set_online_tag(self, tag, recorder: Union[list, object]):
        """
        设置模型的`tag`标记其是否为在线状态。

        参数:
            tag (str): `ONLINE_TAG`或`OFFLINE_TAG`中的标签
            recorder (Union[list,object]): 模型的记录器
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self, recorder: object) -> str:
        """
        给定模型记录器，返回其在线标签。

        参数:
            recorder (Object): 模型的记录器

        返回:
            str: 在线标签
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, recorder: Union[list, object]):
        """
        将所有模型下线并将指定记录器设置为'online'。

        参数:
            recorder (Union[list,object]):
                要重置为'online'的记录器

        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def online_models(self) -> list:
        """
        获取当前`online`模型

        返回:
            list: `online`模型列表
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")

    def update_online_pred(self, to_date=None):
        """
        将`online`模型的预测更新到to_date。

        参数:
            to_date (pd.Timestamp): 更新此日期之前的预测。None表示更新到最新。

        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")


class OnlineToolR(OnlineTool):
    """
    基于记录器(R)的OnlineTool实现。
    """

    def __init__(self, default_exp_name: str = None):
        """
        初始化OnlineToolR。

        参数:
            default_exp_name (str): 默认实验名称
        """
        super().__init__()
        self.default_exp_name = default_exp_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        """
        设置模型记录器的`tag`标记其是否为在线状态。

        参数:
            tag (str): `ONLINE_TAG`、`NEXT_ONLINE_TAG`或`OFFLINE_TAG`中的标签
            recorder (Union[Recorder, List]): 记录器列表或单个记录器实例
        """
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder) -> str:
        """
        给定模型记录器，返回其在线标签。

        参数:
            recorder (Recorder): 记录器实例

        返回:
            str: 在线标签
        """
        tags = recorder.list_tags()
        return tags.get(self.ONLINE_KEY, self.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List], exp_name: str = None):
        """
        将所有模型下线并将指定记录器设置为'online'。

        参数:
            recorder (Union[Recorder, List]):
                要重置为'online'的记录器
            exp_name (str): 实验名称。如果为None则使用default_exp_name

        """
        exp_name = self._get_exp_name(exp_name)
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        recs = list_recorders(exp_name)
        self.set_online_tag(self.OFFLINE_TAG, list(recs.values()))
        self.set_online_tag(self.ONLINE_TAG, recorder)

    def online_models(self, exp_name: str = None) -> list:
        """
        获取当前`online`模型

        参数:
            exp_name (str): 实验名称。如果为None则使用default_exp_name

        返回:
            list: `online`模型列表
        """
        exp_name = self._get_exp_name(exp_name)
        return list(list_recorders(exp_name, lambda rec: self.get_online_tag(rec) == self.ONLINE_TAG).values())

    def update_online_pred(self, to_date=None, from_date=None, exp_name: str = None):
        """
        将在线模型的预测更新到to_date。

        参数:
            to_date (pd.Timestamp): 更新此日期之前的预测。None表示更新到日历中的最新时间
            exp_name (str): 实验名称。如果为None则使用default_exp_name
        """
        exp_name = self._get_exp_name(exp_name)
        online_models = self.online_models(exp_name=exp_name)
        for rec in online_models:
            try:
                updater = PredUpdater(rec, to_date=to_date, from_date=from_date)
            except LoadObjectError as e:
                # 跳过没有预测结果的记录器
                self.logger.warn(f"An exception `{str(e)}` happened when load `pred.pkl`, skip it.")
                continue
            updater.update()

        self.logger.info(f"已完成{exp_name}中{len(online_models)}个在线模型预测的更新。")

    def _get_exp_name(self, exp_name):
        if exp_name is None:
            if self.default_exp_name is None:
                raise ValueError(
                    "default_exp_name和exp_name都为None。OnlineToolR需要一个特定的实验。"
                )
            exp_name = self.default_exp_name
        return exp_name
