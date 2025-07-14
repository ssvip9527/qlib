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
"""实验的默认数据格式化函数。

对于新数据集，继承GenericDataFormatter并实现所有抽象函数。

这些数据集特定的方法：
1) 定义模型使用的表格数据的列和输入类型
2) 执行必要的输入特征工程和归一化步骤
3) 还原预测的归一化
4) 负责训练、验证和测试集的拆分


"""

import abc
import enum


# Type definitions
class DataTypes(enum.IntEnum):
    """定义每列的数值类型。"""

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """定义每列的输入类型。"""

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index


class GenericDataFormatter(abc.ABC):
    """所有数据格式化器的抽象基类。

    用户可以实现以下抽象方法来执行特定于数据集的操作。

    """

    @abc.abstractmethod
    def set_scalers(self, df):
        """使用提供的数据校准缩放器。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_inputs(self, df):
        """执行特征转换。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def format_predictions(self, df):
        """还原任何归一化，使预测结果恢复原始尺度。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def split_data(self, df):
        """执行默认的训练、验证和测试集拆分。"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _column_definition(self):
        """定义每列的顺序、输入类型和数据类型。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_fixed_params(self):
        """定义模型训练时使用的固定参数。

        需要包含以下键：
        'total_time_steps': 定义TFT使用的总时间步数
        'num_encoder_steps': 确定LSTM编码器的长度（即历史数据长度）
        'num_epochs': 训练的最大轮数
        'early_stopping_patience': Keras的早停参数
        'multiprocessing_workers': 数据处理使用的CPU数量


        返回：
            固定参数的字典，例如：

            fixed_params = {
                'total_time_steps': 252 + 5,
                'num_encoder_steps': 252,
                'num_epochs': 100,
                'early_stopping_patience': 5,
                'multiprocessing_workers': 5,
            }
        """
        raise NotImplementedError

    # Shared functions across data-formatters
    @property
    def num_classes_per_cat_input(self):
        """Returns number of categories per relevant input.

        This is seqeuently required for keras embedding layers.
        """
        return self._num_classes_per_cat_input

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.

        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.

        Returns:
          Tuple of (training samples, validation samples)
        """
        return -1, -1

    def get_column_definition(self):
        """Returns formatted column definition in order expected by the TFT."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):
            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError("Illegal number of inputs ({}) of type {}".format(length, input_type))

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.REAL_VALUED and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.CATEGORICAL and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs

    def _get_input_columns(self):
        """Returns names of all input columns."""
        return [tup[0] for tup in self.get_column_definition() if tup[2] not in {InputTypes.ID, InputTypes.TIME}]

    def _get_tft_input_indices(self):
        """Returns the relevant indexes and input sizes required by TFT."""

        # Functions
        def _extract_tuples_from_data_type(data_type, defn):
            return [tup for tup in defn if tup[1] == data_type and tup[2] not in {InputTypes.ID, InputTypes.TIME}]

        def _get_locations(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[2] in input_types]

        # Start extraction
        column_definition = [
            tup for tup in self.get_column_definition() if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        categorical_inputs = _extract_tuples_from_data_type(DataTypes.CATEGORICAL, column_definition)
        real_inputs = _extract_tuples_from_data_type(DataTypes.REAL_VALUED, column_definition)

        locations = {
            "input_size": len(self._get_input_columns()),
            "output_size": len(_get_locations({InputTypes.TARGET}, column_definition)),
            "category_counts": self.num_classes_per_cat_input,
            "input_obs_loc": _get_locations({InputTypes.TARGET}, column_definition),
            "static_input_loc": _get_locations({InputTypes.STATIC_INPUT}, column_definition),
            "known_regular_inputs": _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, real_inputs),
            "known_categorical_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, categorical_inputs
            ),
        }

        return locations

    def get_experiment_params(self):
        """Returns fixed model parameters for experiments."""

        required_keys = [
            "total_time_steps",
            "num_encoder_steps",
            "num_epochs",
            "early_stopping_patience",
            "multiprocessing_workers",
        ]

        fixed_params = self.get_fixed_params()

        for k in required_keys:
            if k not in fixed_params:
                raise ValueError("Field {}".format(k) + " missing from fixed parameter definitions!")

        fixed_params["column_definition"] = self.get_column_definition()

        fixed_params.update(self._get_tft_input_indices())

        return fixed_params
