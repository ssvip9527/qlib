# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权

# pylint: skip-file
# flake8: noqa

import os
import yaml
import json
import copy
import pickle
import logging
import importlib
import subprocess
import pandas as pd
import numpy as np

from abc import abstractmethod

from ...log import get_module_logger, TimeInspector
from hyperopt import fmin, tpe
from hyperopt import STATUS_OK, STATUS_FAIL


class Tuner:
    def __init__(self, tuner_config, optim_config):
        self.logger = get_module_logger("Tuner", sh_level=logging.INFO)

        self.tuner_config = tuner_config
        self.optim_config = optim_config

        self.max_evals = self.tuner_config.get("max_evals", 10)
        self.ex_dir = os.path.join(
            self.tuner_config["experiment"]["dir"],
            self.tuner_config["experiment"]["name"],
        )

        self.best_params = None
        self.best_res = None

        self.space = self.setup_space()

    def tune(self):
        TimeInspector.set_time_mark()
        fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            show_progressbar=False,
        )
        self.logger.info("Local best params: {} ".format(self.best_params))
        TimeInspector.log_cost_time(
            "Finished searching best parameters in Tuner {}.".format(self.tuner_config["experiment"]["id"])
        )

        self.save_local_best_params()

    @abstractmethod
    def objective(self, params):
        """
        实现此方法以使用搜索空间中的参数给出优化因子。
        :return: {'loss': 优化因子，浮点型,
                  'status': 此评估步骤的状态，STATUS_OK或STATUS_FAIL}。
        """
        pass

    @abstractmethod
    def setup_space(self):
        """
        实现此方法以设置调优器的搜索空间。
        :return: 搜索空间，字典类型。
        """
        pass

    @abstractmethod
    def save_local_best_params(self):
        """
        实现此方法以保存此调优器的最佳参数。
        """
        pass


class QLibTuner(Tuner):
    ESTIMATOR_CONFIG_NAME = "estimator_config.yaml"
    EXP_INFO_NAME = "exp_info.json"
    EXP_RESULT_DIR = "sacred/{}"
    EXP_RESULT_NAME = "analysis.pkl"
    LOCAL_BEST_PARAMS_NAME = "local_best_params.json"

    def objective(self, params):
        # 1. 为特定的评估器进程设置配置
        estimator_path = self.setup_estimator_config(params)
        self.logger.info("搜索参数: {} ".format(params))

        # 2. 使用子进程执行评估器程序，此进程将等待直到子进程完成
        sub_fails = subprocess.call("estimator -c {}".format(estimator_path), shell=True)
        if sub_fails:
            # 如果子进程失败，忽略此评估步骤
            self.logger.info("使用这些搜索参数时评估器实验失败")
            return {"loss": np.nan, "status": STATUS_FAIL}

        # 3. 获取子进程的结果，并检查结果是否为Nan
        res = self.fetch_result()
        if np.isnan(res):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        # 4. 保存最佳分数和参数
        if self.best_res is None or self.best_res > res:
            self.best_res = res
            self.best_params = params

        # 5. 返回结果作为优化目标
        return {"loss": res, "status": status}

    def fetch_result(self):
        # 1. Get experiment information
        exp_info_path = os.path.join(self.ex_dir, QLibTuner.EXP_INFO_NAME)
        with open(exp_info_path) as fp:
            exp_info = json.load(fp)
        estimator_ex_id = exp_info["id"]

        # 2. Return model result if needed
        if self.optim_config.report_type == "model":
            if self.optim_config.report_factor == "model_score":
                # if estimator experiment is multi-label training, user need to process the scores by himself
                # Default method is return the average score
                return np.mean(exp_info["performance"]["model_score"])
            elif self.optim_config.report_factor == "model_pearsonr":
                # pearsonr is a correlation coefficient, 1 is the best
                return np.abs(exp_info["performance"]["model_pearsonr"] - 1)

        # 3. Get backtest results
        exp_result_dir = os.path.join(self.ex_dir, QLibTuner.EXP_RESULT_DIR.format(estimator_ex_id))
        exp_result_path = os.path.join(exp_result_dir, QLibTuner.EXP_RESULT_NAME)
        with open(exp_result_path, "rb") as fp:
            analysis_df = pickle.load(fp)

        # 4. Get the backtest factor which user want to optimize, if user want to maximize the factor, then reverse the result
        res = analysis_df.loc[self.optim_config.report_type].loc[self.optim_config.report_factor]
        # res = res.values[0] if self.optim_config.optim_type == 'min' else -res.values[0]
        if self.optim_config == "min":
            return res.values[0]
        elif self.optim_config == "max":
            return -res.values[0]
        else:
            # self.optim_config == 'correlation'
            return np.abs(res.values[0] - 1)

    def setup_estimator_config(self, params):
        estimator_config = copy.deepcopy(self.tuner_config)
        estimator_config["model"].update({"args": params["model_space"]})
        estimator_config["strategy"].update({"args": params["strategy_space"]})
        if params.get("data_label_space", None) is not None:
            estimator_config["data"]["args"].update(params["data_label_space"])

        estimator_path = os.path.join(
            self.tuner_config["experiment"].get("dir", "../"),
            QLibTuner.ESTIMATOR_CONFIG_NAME,
        )

        with open(estimator_path, "w") as fp:
            yaml.dump(estimator_config, fp)

        return estimator_path

    def setup_space(self):
        # 1. Setup model space
        model_space_name = self.tuner_config["model"].get("space", None)
        if model_space_name is None:
            raise ValueError("Please give the search space of model.")
        model_space = getattr(
            importlib.import_module(".space", package="qlib.contrib.tuner"),
            model_space_name,
        )

        # 2. Setup strategy space
        strategy_space_name = self.tuner_config["strategy"].get("space", None)
        if strategy_space_name is None:
            raise ValueError("Please give the search space of strategy.")
        strategy_space = getattr(
            importlib.import_module(".space", package="qlib.contrib.tuner"),
            strategy_space_name,
        )

        # 3. Setup data label space if given
        if self.tuner_config.get("data_label", None) is not None:
            data_label_space_name = self.tuner_config["data_label"].get("space", None)
            if data_label_space_name is not None:
                data_label_space = getattr(
                    importlib.import_module(".space", package="qlib.contrib.tuner"),
                    data_label_space_name,
                )
        else:
            data_label_space_name = None

        # 4. Combine the searching space
        space = dict()
        space.update({"model_space": model_space})
        space.update({"strategy_space": strategy_space})
        if data_label_space_name is not None:
            space.update({"data_label_space": data_label_space})

        return space

    def save_local_best_params(self):
        TimeInspector.set_time_mark()
        local_best_params_path = os.path.join(self.ex_dir, QLibTuner.LOCAL_BEST_PARAMS_NAME)
        with open(local_best_params_path, "w") as fp:
            json.dump(self.best_params, fp)
        TimeInspector.log_cost_time(
            "Finished saving local best tuner parameters to: {} .".format(local_best_params_path)
        )
