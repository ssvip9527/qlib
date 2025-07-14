# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
本示例展示了当需要更新预测时OnlineTool的工作方式。
包含两部分：首次训练和更新在线预测。
首先，我们将完成训练并将训练好的模型设置为`在线`模型。
接下来，我们将完成在线预测的更新。
"""
import copy
import fire
import qlib
from qlib.constant import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.online.utils import OnlineToolR
from qlib.tests.config import CSI300_GBDT_TASK

task = copy.deepcopy(CSI300_GBDT_TASK)

task["record"] = {
    "class": "SignalRecord",
    "module_path": "qlib.workflow.record_temp",
}


class UpdatePredExample:
    def __init__(
        self, provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, experiment_name="online_srv", task_config=task
    ):
        qlib.init(provider_uri=provider_uri, region=region)
        self.experiment_name = experiment_name
        self.online_tool = OnlineToolR(self.experiment_name)
        self.task_config = task_config

    def first_train(self):
        rec = task_train(self.task_config, experiment_name=self.experiment_name)
        self.online_tool.reset_online_tag(rec)  # set to online model

    def update_online_pred(self):
        self.online_tool.update_online_pred()

    def main(self):
        self.first_train()
        self.update_online_pred()


if __name__ == "__main__":
    ## 要训练模型并将其设置为在线模型，请使用以下命令
    # python update_online_pred.py first_train
    ## 要每天更新一次在线预测，请使用以下命令
    # python update_online_pred.py update_online_pred
    ## 要使用自定义参数查看整个流程，请使用以下命令
    # python update_online_pred.py main --experiment_name="your_exp_name"
    fire.Fire(UpdatePredExample)
