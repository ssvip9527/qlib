# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
本示例展示如何基于滚动任务模拟OnlineManager。
"""

from pprint import pprint
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG_ONLINE, CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE
import pandas as pd
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


class OnlineSimulationExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        exp_name="rolling_exp",
        task_url="mongodb://10.0.0.4:27017/",  # not necessary when using TrainerR or DelayTrainerR
        task_db_name="rolling_db",  # not necessary when using TrainerR or DelayTrainerR
        task_pool="rolling_task",
        rolling_step=80,
        start_time="2018-09-10",
        end_time="2018-10-31",
        tasks=None,
        trainer="TrainerR",
    ):
        """
        初始化OnlineManagerExample。

        Args:
            provider_uri (str, optional): 数据提供器URI。默认为 "~/.qlib/qlib_data/cn_data".
            region (str, optional): 股票市场区域。默认为 "cn".
            exp_name (str, optional): 实验名称。默认为 "rolling_exp".
            task_url (str, optional): MongoDB连接URL。默认为 "mongodb://10.0.0.4:27017/".
            task_db_name (str, optional): 数据库名称。默认为 "rolling_db".
            task_pool (str, optional): 任务池名称（任务池是MongoDB中的一个集合）。默认为 "rolling_task".
            rolling_step (int, optional): 滚动窗口步长。默认为80.
            start_time (str, optional): 模拟开始时间。默认为 "2018-09-10".
            end_time (str, optional): 模拟结束时间。默认为 "2018-10-31".
            tasks (dict or list[dict]): 等待滚动训练的任务配置集合
        """
        if tasks is None:
            tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE, CSI100_RECORD_LGB_TASK_CONFIG_ONLINE]
        self.exp_name = exp_name
        self.task_pool = task_pool
        self.start_time = start_time
        self.end_time = end_time
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.rolling_gen = RollingGen(
            step=rolling_step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None
        )  # The rolling tasks generator, ds_extra_mod_func is None because we just need to simulate to 2018-10-31 and needn't change the handler end time.
        if trainer == "TrainerRM":
            self.trainer = TrainerRM(self.exp_name, self.task_pool)
        elif trainer == "TrainerR":
            self.trainer = TrainerR(self.exp_name)
        else:
            # TODO: support all the trainers: TrainerR, TrainerRM, DelayTrainerR
            raise NotImplementedError(f"This type of input is not supported")
        self.rolling_online_manager = OnlineManager(
            RollingStrategy(exp_name, task_template=tasks, rolling_gen=self.rolling_gen),
            trainer=self.trainer,
            begin_time=self.start_time,
        )
        self.tasks = tasks

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        if isinstance(self.trainer, TrainerRM):
            TaskManager(self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    # Run this to run all workflow automatically
    def main(self):
        print("========== reset ==========")
        self.reset()
        print("========== simulate ==========")
        self.rolling_online_manager.simulate(end_time=self.end_time)
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        signals = self.rolling_online_manager.get_signals()
        print(signals)
        # Backtesting
        # - the code is based on this example https://qlib.moujue.com/component/strategy.html
        CSI300_BENCH = "SH000903"
        STRATEGY_CONFIG = {
            "topk": 30,
            "n_drop": 3,
            "signal": signals.to_frame("score"),
        }
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time=signals.index.get_level_values("datetime").min(),
            end_time=signals.index.get_level_values("datetime").max(),
            strategy=strategy_obj,
        )
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)

    def worker(self):
        # train tasks by other progress or machines for multiprocessing
        # FIXME: only can call after finishing simulation when using DelayTrainerRM, or there will be some exception.
        print("========== worker ==========")
        if isinstance(self.trainer, TrainerRM):
            self.trainer.worker()
        else:
            print(f"{type(self.trainer)} is not supported for worker.")


if __name__ == "__main__":
    ## to run all workflow automatically with your own parameters, use the command below
    # python online_management_simulate.py main --experiment_name="your_exp_name" --rolling_step=60
    fire.Fire(OnlineSimulationExample)
