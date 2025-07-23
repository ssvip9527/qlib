# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
本示例展示了OnlineManager如何与滚动任务配合工作。
包含四个部分：首次训练、常规流程1、添加策略和常规流程2。
首先，OnlineManager将完成首次训练并将训练好的模型设置为`在线`模型。
接下来，OnlineManager将完成常规流程，包括更新在线预测 -> 准备任务 -> 准备新模型 -> 准备信号
然后，我们将向OnlineManager添加一些新策略。这将完成新策略的首次训练。
最后，OnlineManager将完成第二次常规流程并更新所有策略。
"""

import os
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM, end_task_train, task_train
from qlib.workflow import R
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.online.manager import OnlineManager
from qlib.tests.config import CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING, CSI100_RECORD_LGB_TASK_CONFIG_ROLLING
from qlib.workflow.task.manage import TaskManager


class RollingOnlineExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        trainer=DelayTrainerRM(),  # 你可以从 TrainerR、TrainerRM、DelayTrainerR、DelayTrainerRM 中选择
        task_url="mongodb://10.0.0.4:27017/",  # 使用 TrainerR 或 DelayTrainerR 时不需要
        task_db_name="rolling_db",  # 使用 TrainerR 或 DelayTrainerR 时不需要
        rolling_step=550,
        tasks=None,
        add_tasks=None,
    ):
        if add_tasks is None:
            add_tasks = [CSI100_RECORD_LGB_TASK_CONFIG_ROLLING]
        if tasks is None:
            tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING]
        mongo_conf = {
            "task_url": task_url,  # 你的 MongoDB url
            "task_db_name": task_db_name,  # 数据库名称
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.tasks = tasks
        self.add_tasks = add_tasks
        self.rolling_step = rolling_step
        strategies = []
        for task in tasks:
            name_id = task["model"]["class"]  # 注意：假设：模型类只能指定一个策略
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.trainer = trainer
        self.rolling_online_manager = OnlineManager(strategies, trainer=self.trainer)

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # OnlineManager 会将数据转储到此文件，以便在调用例程时加载。
    )

    def worker(self):
        # 通过其他进程或机器训练任务以实现多进程处理
        print("========== worker ==========")
        if isinstance(self.trainer, TrainerRM):
            for task in self.tasks + self.add_tasks:
                name_id = task["model"]["class"]
                self.trainer.worker(experiment_name=name_id)
        else:
            print(f"{type(self.trainer)} is not supported for worker.")

    # 将所有内容重置为初始状态，保存重要数据时需谨慎
    def reset(self):
        for task in self.tasks + self.add_tasks:
            name_id = task["model"]["class"]
            TaskManager(task_pool=name_id).remove()
            exp = R.get_exp(experiment_name=name_id)
            for rid in exp.list_recorders():
                exp.delete_recorder(rid)

        if os.path.exists(self._ROLLING_MANAGER_PATH):
            os.remove(self._ROLLING_MANAGER_PATH)

    def first_run(self):
        print("========== reset ==========")
        self.reset()
        print("========== first_run ==========")
        self.rolling_online_manager.first_train()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def routine(self):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== routine ==========")
        self.rolling_online_manager.routine()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        print(self.rolling_online_manager.get_signals())
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def add_strategy(self):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== add strategy ==========")
        strategies = []
        for task in self.add_tasks:
            name_id = task["model"]["class"]  # NOTE: Assumption: The model class can specify only one strategy
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=self.rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.rolling_online_manager.add_strategy(strategies=strategies)
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def main(self):
        self.first_run()
        self.routine()
        self.add_strategy()
        self.routine()


if __name__ == "__main__":
    ####### 要训练第一版模型，请使用以下命令
    # python rolling_online_management.py first_run

    ####### 要在交易时间后更新模型和预测，请使用以下命令
    # python rolling_online_management.py routine

    ####### 要定义自己的参数，请使用`--`
    # python rolling_online_management.py first_run --exp_name='your_exp_name' --rolling_step=40
    fire.Fire(RollingOnlineExample)
