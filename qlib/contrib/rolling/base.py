# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from copy import deepcopy
from pathlib import Path
from ruamel.yaml import YAML
from typing import List, Optional, Union

import fire
import pandas as pd

from qlib import auto_init
from qlib.log import get_module_logger
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.model.trainer import TrainerR
from qlib.utils import get_cls_kwargs, init_instance_by_config
from qlib.utils.data import update_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import replace_task_handler_with_cache


class Rolling:
    """
    滚动模块的动机
    - 它仅专注于**离线**将特定任务转换为滚动任务
    - 为简化实现，忽略了以下因素：
        - 任务之间的依赖性（例如时间序列）

    相关模块及与本模块的区别：
    - MetaController：它学习如何处理任务（例如学习如何学习）。
        - 但滚动模块关注的是如何将单个任务拆分为时间序列中的多个任务并运行它们。
    - OnlineStrategy：它专注于模型服务，可以随时间更新模型。
        - 滚动模块更简单，仅用于离线测试滚动模型。它不希望与OnlineStrategy共享接口。

    滚动相关的代码在`task_generator`和`RollingGen`级别与上述模块共享
    但由于用途不同，其他部分不共享。


    .. code-block:: shell

        # 以下是本模块的典型使用示例
        python -m qlib.contrib.rolling.base --conf_path <yaml文件路径> run

    **注意**
    运行示例前，请使用以下命令清理之前的结果：
    - `rm -r mlruns`
    - 因为很难永久删除实验（它会被移到.trash目录，并在创建同名实验时引发错误）。

    """

    def __init__(
        self,
        conf_path: Union[str, Path],
        exp_name: Optional[str] = None,
        horizon: Optional[int] = 20,
        step: int = 20,
        h_path: Optional[str] = None,
        train_start: Optional[str] = None,
        test_end: Optional[str] = None,
        task_ext_conf: Optional[dict] = None,
        rolling_exp: Optional[str] = None,
    ) -> None:
        """
        参数
        ----------
        conf_path : str
            滚动配置的路径。
        exp_name : Optional[str]
            输出的实验名称（输出是一个包含连接的滚动记录预测的记录）。
        horizon: Optional[int] = 20,
            预测目标的时间跨度。
            用于覆盖文件中的预测时间跨度。
        h_path : Optional[str]
            这是作为处理器导出的其他数据源。它将覆盖配置中的数据处理器部分。
            如果未提供，当 `enable_handler_cache=True` 时将为处理器创建自定义缓存。
        test_end : Optional[str]
            数据的测试结束时间。通常与处理器一起使用。
            你也可以通过 task_ext_conf 以更复杂的方式实现相同的功能。
        train_start : Optional[str]
            数据的训练开始时间。通常与处理器一起使用。
            你也可以通过 task_ext_conf 以更复杂的方式实现相同的功能。
        task_ext_conf : Optional[dict]
            用于更新任务配置的选项。
        rolling_exp : Optional[str]
            滚动实验的名称。
            它将在一个实验中包含许多记录。每个记录对应一个特定的滚动。
            请注意，这与最终实验不同。
        """
        self.logger = get_module_logger("Rolling")
        self.conf_path = Path(conf_path)
        self.exp_name = exp_name
        self._rid = None  # the final combined recorder id in `exp_name`

        self.step = step
        assert horizon is not None, "Current version does not support extracting horizon from the underlying dataset"
        self.horizon = horizon
        if rolling_exp is None:
            datetime_suffix = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            self.rolling_exp = f"rolling_models_{datetime_suffix}"
        else:
            self.rolling_exp = rolling_exp
            self.logger.warning(
                "Using user specifiied name for rolling models. So the experiment names duplicateds. "
                "Please manually remove your experiment for rolling model with command like `rm -r mlruns`."
                " Otherwise it will prevents the creating of experimen with same name"
            )
        self.train_start = train_start
        self.test_end = test_end
        self.task_ext_conf = task_ext_conf
        self.h_path = h_path

        # FIXME:
        # - the qlib_init section will be ignored by me.
        # - So we have to design a priority mechanism to solve this issue.

    def _raw_conf(self) -> dict:
        with self.conf_path.open("r") as f:
            yaml = YAML(typ="safe", pure=True)
            return yaml.load(f)

    def _replace_handler_with_cache(self, task: dict):
        """
        由于原始滚动中的数据处理部分较慢。所以我们必须
        这个类尝试添加更多功能
        """
        if self.h_path is not None:
            h_path = Path(self.h_path)
            task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        else:
            task = replace_task_handler_with_cache(task, self.conf_path.parent)
        return task

    def _update_start_end_time(self, task: dict):
        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = pd.Timestamp(self.train_start), seg[1]

        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.test_end)
        return task

    def basic_task(self, enable_handler_cache: Optional[bool] = True):
        """
        基本任务可能与 __init__ 中的 `conf_path` 配置不完全相同，原因如下：
        - 一些参数可能被 __init__ 中的参数覆盖
        - 用户可以实现子类来提高性能
        """
        task: dict = self._raw_conf()["task"]
        task = deepcopy(task)

        # modify dataset horizon
        # NOTE:
        # It assumpts that the label can be modifiled in the handler's kwargs
        # But is not always a valid. It is only valid in the predefined dataset `Alpha158` & `Alpha360`
        if self.horizon is None:
            # TODO:
            # - get horizon automatically from the expression!!!!
            raise NotImplementedError(f"This type of input is not supported")
        else:
            if enable_handler_cache and self.h_path is not None:
                self.logger.info("Fail to override the horizon due to data handler cache")
            else:
                self.logger.info("The prediction horizon is overrided")
                if isinstance(task["dataset"]["kwargs"]["handler"], dict):
                    task["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                        "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
                    ]
                else:
                    self.logger.warning("Try to automatically configure the lablel but failed.")

        if self.h_path is not None or enable_handler_cache:
            # if we already have provided data source or we want to create one
            task = self._replace_handler_with_cache(task)
        task = self._update_start_end_time(task)

        if self.task_ext_conf is not None:
            task = update_config(task, self.task_ext_conf)
        self.logger.info(task)
        return task

    def run_basic_task(self):
        """
        运行不带滚动的基本任务。
        这用于快速测试模型调优。
        """
        task = self.basic_task()
        print(task)
        trainer = TrainerR(experiment_name=self.exp_name)
        trainer([task])

    def get_task_list(self) -> List[dict]:
        """返回一批用于滚动的任务。"""
        task = self.basic_task()
        task_l = task_generator(
            task, RollingGen(step=self.step, trunc_days=self.horizon + 1)
        )  # the last two days should be truncated to avoid information leakage
        for t in task_l:
            # 当我们滚动任务时，不需要进一步分析。
            # 分析推迟到最终集成。
            t["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task_l

    def _train_rolling_tasks(self):
        task_l = self.get_task_list()
        self.logger.info("Deleting previous Rolling results")
        try:
            # TODO: mlflow 不支持永久删除实验
            # 它将被移动到 .trash 并阻止创建同名实验
            R.delete_exp(experiment_name=self.rolling_exp)  # We should remove the rolling experiments.
        except ValueError:
            self.logger.info("No previous rolling results")
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    def _ens_rolling(self):
        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()
        with R.start(experiment_name=self.exp_name):
            R.log_params(exp_name=self.rolling_exp)
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
            self._rid = R.get_recorder().id

    def _update_rolling_rec(self):
        """
        评估组合的滚动结果
        """
        rec = R.get_recorder(experiment_name=self.exp_name, recorder_id=self._rid)
        # Follow the original analyser
        records = self._raw_conf()["task"].get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            if issubclass(get_cls_kwargs(record)[0], SignalRecord):
                # 跳过信号记录。
                continue
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_temp",
            )
            r.generate()
        print(f"Your evaluation results can be found in the experiment named `{self.exp_name}`.")

    def run(self):
        # # 结果将保存在 mlruns 中。
        # 1) 每个滚动任务保存在 rolling_models 中
        # 2) 组合的滚动任务和评估结果保存在 rolling 中
        self._ens_rolling()
        self._update_rolling_rec()


if __name__ == "__main__":
    auto_init()
    fire.Fire(Rolling)
