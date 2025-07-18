# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权

# pylint: skip-file
# flake8: noqa

# coding=utf-8

import argparse
import importlib
import os
import yaml

from .config import TunerConfigManager


args_parser = argparse.ArgumentParser(prog="tuner")
args_parser.add_argument(
    "-c",
    "--config_path",
    required=True,
    type=str,
    help="配置文件路径，指定从何处加载YAML配置。",
)

args = args_parser.parse_args()

TUNER_CONFIG_MANAGER = TunerConfigManager(args.config_path)


def run():
    # 1. Get pipeline class.
    tuner_pipeline_class = getattr(importlib.import_module(".pipeline", package="qlib.contrib.tuner"), "Pipeline")
    # 2. Init tuner pipeline.
    tuner_pipeline = tuner_pipeline_class(TUNER_CONFIG_MANAGER)
    # 3. Begin to tune
    tuner_pipeline.run()
