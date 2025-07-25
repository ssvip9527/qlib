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
"""Alpha158数据集的自定义格式化函数。

定义数据集特定的列定义和数据转换。
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class Alpha158Formatter(GenericDataFormatter):
    """为Alpha158数据集定义和格式化数据。

    属性：
        column_definition: 定义实验中使用的列的输入和数据类型。
        identifiers: 实验中使用的实体标识符。
"""

    _column_definition = [
        ("instrument", DataTypes.CATEGORICAL, InputTypes.ID),
        ("LABEL0", DataTypes.REAL_VALUED, InputTypes.TARGET),
        ("date", DataTypes.DATE, InputTypes.TIME),
        ("month", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ("day_of_week", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # Selected features
        ("RESI5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("WVMA5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("KLEN", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORD5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("ROC60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RESI10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("VSTD5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("WVMA60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("STD5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR20", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORD60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORD10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR20", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("KLOW", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("const", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """初始化格式化器。"""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, valid_boundary=2016, test_boundary=2018):
        """将数据框拆分为训练-验证-测试数据框。

        这也会校准缩放对象，并为每个拆分转换数据。

        参数：
            df: 要拆分的源数据框。
            valid_boundary: 验证数据的起始年份
            test_boundary: 测试数据的起始年份

        返回：
            转换后的（训练、验证、测试）数据元组。
        """

        print("Formatting train-valid-test splits.")

        index = df["year"]
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[index >= test_boundary]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """使用提供的数据校准缩放器。

        参数：
            df: 用于校准缩放器的数据。
        """
        print("Setting scalers with training data...")

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values
        )  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """执行特征转换。

        这包括特征工程、预处理和归一化。

        参数：
            df: 要转换的数据框。

        返回：
            转换后的数据框。

        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """还原任何归一化，使预测结果恢复原始尺度。

        参数：
            predictions: 模型预测的数据框。

        返回：
            未归一化预测的数据框。
        """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {"forecast_time", "identifier"}:
                # Using [col] is for aligning with the format when fitting
                output[col] = self._target_scaler.inverse_transform(predictions[[col]])

        return output

    # Default params
    def get_fixed_params(self):
        """返回实验的固定模型参数。"""

        fixed_params = {
            "total_time_steps": 6 + 6,
            "num_encoder_steps": 6,
            "num_epochs": 100,
            "early_stopping_patience": 10,
            "multiprocessing_workers": 5,
        }

        return fixed_params

    def get_default_model_params(self):
        """返回默认优化的模型参数。"""

        model_params = {
            "dropout_rate": 0.4,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "minibatch_size": 128,
            "max_gradient_norm": 0.0135,
            "num_heads": 1,
            "stack_size": 1,
        }

        return model_params
