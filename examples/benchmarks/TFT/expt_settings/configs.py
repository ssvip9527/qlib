# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""TFT实验的默认配置。

包含数据、序列化模型和预测的默认输出路径，用于出版物中的主要实验。
"""

import os

import data_formatters.qlib_Alpha158


class ExperimentConfig:
    """定义实验配置和输出路径。

    属性：
        root_folder: 包含所有实验输出的根文件夹。
        experiment: 要运行的实验名称。
        data_folder: 存储实验数据的文件夹。
        model_folder: 存储序列化模型的文件夹。
        results_folder: 存储结果的文件夹。
        data_csv_path: 实验中使用的主要数据CSV文件路径。
        hyperparam_iterations: 实验的默认随机搜索迭代次数。
    """

    default_experiments = ["Alpha158"]

    def __init__(self, experiment="volatility", root_folder=None):
        """基于选择的默认实验创建配置。

        参数：
            experiment: 实验名称。
            root_folder: 保存所有训练输出的根文件夹。
        """

        if experiment not in self.default_experiments:
            raise ValueError("无法识别的实验={}".format(experiment))

        # Defines all relevant paths
        if root_folder is None:
            root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "outputs")
            print("Using root folder {}".format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, "data", experiment)
        self.model_folder = os.path.join(root_folder, "saved_models", experiment)
        self.results_folder = os.path.join(root_folder, "results", experiment)

        # Creates folders if they don't exist
        for relevant_directory in [self.root_folder, self.data_folder, self.model_folder, self.results_folder]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            "Alpha158": "Alpha158.csv",
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    @property
    def hyperparam_iterations(self):
        return 240 if self.experiment == "volatility" else 60

    def make_data_formatter(self):
        """获取实验的数据格式化器对象。

        返回：
            每个实验的默认DataFormatter。
        """

        data_formatter_class = {
            "Alpha158": data_formatters.qlib_Alpha158.Alpha158Formatter,
        }

        return data_formatter_class[self.experiment]()
