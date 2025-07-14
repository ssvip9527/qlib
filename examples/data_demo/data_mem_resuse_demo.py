# 版权所有 (c) 微软公司。
# 根据 MIT 许可证授权。
"""
此演示的目的
- 展示 Qlib 的数据模块是可序列化的，用户可以将处理后的数据转储到磁盘以避免重复的数据预处理
"""

from copy import deepcopy
from pathlib import Path
import pickle
from pprint import pprint
from ruamel.yaml import YAML
import subprocess

from qlib import init
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import TimeInspector
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config

# 为通用目的，我们使用相对路径
DIRNAME = Path(__file__).absolute().resolve().parent

if __name__ == "__main__":
    init()

    repeat = 2
    exp_name = "data_mem_reuse_demo"

    config_path = DIRNAME.parent / "benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"
    yaml = YAML(typ="safe", pure=True)
    task_config = yaml.load(config_path.open())

    # 1) 不使用内存中已处理的数据
    with TimeInspector.logt("不重用内存中已处理数据的原始时间:"):
        for i in range(repeat):
            task_train(task_config["task"], experiment_name=exp_name)

    # 2) 在内存中准备已处理的数据。
    hd_conf = task_config["task"]["dataset"]["kwargs"]["handler"]
    pprint(hd_conf)
    hd: DataHandlerLP = init_instance_by_config(hd_conf)

    # 3) 重用内存中已处理的数据
    new_task = deepcopy(task_config["task"])
    new_task["dataset"]["kwargs"]["handler"] = hd
    print(new_task)

    with TimeInspector.logt("重用内存中已处理数据的时间:"):
        # 这将节省从磁盘重新加载和处理数据的时间（在`DataHandlerLP`中）
        # 回测阶段仍然需要花费大量时间
        for i in range(repeat):
            task_train(new_task, experiment_name=exp_name)

    # 4) 用户可以修改除内存中已处理数据（处理器）之外的其他部分
    new_task = deepcopy(task_config["task"])
    new_task["dataset"]["kwargs"]["segments"]["train"] = ("20100101", "20131231")
    with TimeInspector.logt("重用内存中已处理数据的时间:"):
        task_train(new_task, experiment_name=exp_name)
