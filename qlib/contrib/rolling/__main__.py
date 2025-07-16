import fire
from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.utils.mod import find_all_classes

if __name__ == "__main__":
    sub_commands = {}
    for cls in find_all_classes("qlib.contrib.rolling", Rolling):
        sub_commands[cls.__module__.split(".")[-1]] = cls
    # The sub_commands will be like
    # {'base': <class 'qlib.contrib.rolling.base.Rolling'>, ...}
    # 因此你可以使用以下命令运行
    # - `python -m qlib.contrib.rolling base --conf_path <yaml文件路径> run`
    # - base可以替换为其他模块名称
    auto_init()
    fire.Fire(sub_commands)
