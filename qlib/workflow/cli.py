#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import logging
import os
from pathlib import Path
import sys

import fire
from jinja2 import Template, meta
from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.log import get_module_logger
from qlib.model.trainer import task_train
from qlib.utils import set_log_with_config
from qlib.utils.data import update_config

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)


def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)


def sys_config(config, config_path):
    """
    配置`sys`部分

    参数
    ----------
    config : dict
        工作流配置
    config_path : str
        配置文件路径
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))


def render_template(config_path: str) -> str:
    """
    根据环境渲染模板

    参数
    ----------
    config_path : str
        配置文件路径

    返回
    -------
    str
        渲染后的内容
    """
    with open(config_path, "r") as f:
        config = f.read()
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)

    # Get context from os.environ according to the variables
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    logger.info(f"Render the template with the context: {context}")

    # Render the template with the context
    rendered_content = template.render(context)
    return rendered_content


# workflow handler function
def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    """
    Qlib命令行入口。
    用户可以通过配置文件运行完整的量化研究工作流
    - 代码位于``qlib/workflow/cli.py`

    用户可以在workflow.yml文件中通过添加"BASE_CONFIG_PATH"指定基础配置文件。
    Qlib会先加载BASE_CONFIG_PATH中的配置，用户只需在自己的workflow.yml文件中更新自定义字段。

    示例:

        qlib_init:
            provider_uri: "~/.qlib/qlib_data/cn_data"
            region: cn
        BASE_CONFIG_PATH: "workflow_config_lightgbm_Alpha158_csi500.yaml"
        market: csi300

    """
    # Render the template
    rendered_yaml = render_template(config_path)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        with open(path) as fp:
            yaml = YAML(typ="safe", pure=True)
            base_config = yaml.load(fp)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)


# function to run workflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()
