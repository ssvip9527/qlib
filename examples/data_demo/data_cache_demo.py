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
from qlib.log import TimeInspector

from qlib import init
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

# 为通用目的，我们使用相对路径
DIRNAME = Path(__file__).absolute().resolve().parent

if __name__ == "__main__":
    init()

    config_path = DIRNAME.parent / "benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"

    # 1) 显示原始时间
    with TimeInspector.logt("不使用处理器缓存的原始时间:"):
        subprocess.run(f"qrun {config_path}", shell=True)

    # 2) 转储处理器
    yaml = YAML(typ="safe", pure=True)
    task_config = yaml.load(config_path.open())
    hd_conf = task_config["task"]["dataset"]["kwargs"]["handler"]
    pprint(hd_conf)
    hd: DataHandlerLP = init_instance_by_config(hd_conf)
    hd_path = DIRNAME / "handler.pkl"
    hd.to_pickle(hd_path, dump_all=True)

    # 3) 使用处理器缓存创建新任务
    new_task_config = deepcopy(task_config)
    new_task_config["task"]["dataset"]["kwargs"]["handler"] = f"file://{hd_path}"
    new_task_config["sys"] = {"path": [str(config_path.parent.resolve())]}
    new_task_path = DIRNAME / "new_task.yaml"
    print("新任务的位置", new_task_path)

    # save new task
    with new_task_path.open("w") as f:
        yaml.safe_dump(new_task_config, f, indent=4, sort_keys=False)

    # 4) 使用新任务训练模型
    with TimeInspector.logt("使用处理器缓存的任务时间:"):
        subprocess.run(f"qrun {new_task_path}", shell=True)
