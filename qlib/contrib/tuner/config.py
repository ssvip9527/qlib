# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import copy
import os
from ruamel.yaml import YAML


class TunerConfigManager:
    def __init__(self, config_path):
        if not config_path:
            raise ValueError("配置路径无效。")
        self.config_path = config_path

        with open(config_path) as fp:
            yaml = YAML(typ="safe", pure=True)
            config = yaml.load(fp)
        self.config = copy.deepcopy(config)

        self.pipeline_ex_config = PipelineExperimentConfig(config.get("experiment", dict()), self)
        self.pipeline_config = config.get("tuner_pipeline", list())
        self.optim_config = OptimizationConfig(config.get("optimization_criteria", dict()), self)

        self.time_config = config.get("time_period", dict())
        self.data_config = config.get("data", dict())
        self.backtest_config = config.get("backtest", dict())
        self.qlib_client_config = config.get("qlib_client", dict())


class PipelineExperimentConfig:
    def __init__(self, config, TUNER_CONFIG_MANAGER):
        """
        :param config:  调优实验的配置字典
        :param TUNER_CONFIG_MANAGER:   调优配置管理器
        """
        self.name = config.get("name", "tuner_experiment")
        # 配置文件的目录
        self.global_dir = config.get("dir", os.path.dirname(TUNER_CONFIG_MANAGER.config_path))
        # 调优实验结果的目录
        self.tuner_ex_dir = config.get("tuner_ex_dir", os.path.join(self.global_dir, self.name))
        if not os.path.exists(self.tuner_ex_dir):
            os.makedirs(self.tuner_ex_dir)
        # 所有评估器实验结果的目录
        self.estimator_ex_dir = config.get("estimator_ex_dir", os.path.join(self.tuner_ex_dir, "estimator_experiment"))
        if not os.path.exists(self.estimator_ex_dir):
            os.makedirs(self.estimator_ex_dir)
        # 获取调优器类型
        self.tuner_module_path = config.get("tuner_module_path", "qlib.contrib.tuner.tuner")
        self.tuner_class = config.get("tuner_class", "QLibTuner")
        # 保存调优实验以便进一步查看
        tuner_ex_config_path = os.path.join(self.tuner_ex_dir, "tuner_config.yaml")
        with open(tuner_ex_config_path, "w") as fp:
            yaml.dump(TUNER_CONFIG_MANAGER.config, fp)


class OptimizationConfig:
    def __init__(self, config, TUNER_CONFIG_MANAGER):
        self.report_type = config.get("report_type", "pred_long")
        if self.report_type not in [
            "pred_long",
            "pred_long_short",
            "pred_short",
            "excess_return_without_cost",
            "excess_return_with_cost",
            "model",
        ]:
            raise ValueError(
                "report_type必须是pred_long、pred_long_short、pred_short、excess_return_without_cost、excess_return_with_cost或model中的一种"
            )

        self.report_factor = config.get("report_factor", "information_ratio")
        if self.report_factor not in [
            "annualized_return",
            "information_ratio",
            "max_drawdown",
            "mean",
            "std",
            "model_score",
            "model_pearsonr",
        ]:
            raise ValueError(
                "report_factor必须是annualized_return、information_ratio、max_drawdown、mean、std、model_pearsonr或model_score中的一种"
            )

        self.optim_type = config.get("optim_type", "max")
        if self.optim_type not in ["min", "max", "correlation"]:
            raise ValueError("optim_type必须是min、max或correlation")
