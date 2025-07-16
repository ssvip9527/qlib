# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Text, Union
from catboost import Pool, CatBoost
from catboost.utils import get_gpu_device_count

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter


class CatBoostModel(Model, FeatureInt):
    """CatBoost模型"""

    def __init__(self, loss="RMSE", **kwargs):
        # 还有更多选项
        if loss not in {"RMSE", "Logloss"}:
            raise NotImplementedError
        self._params = {"loss_function": loss}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("数据集数据为空，请检查您的数据集配置。")
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # CatBoost需要一维数组作为标签
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("CatBoost不支持多标签训练")

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train).values
            w_valid = reweighter.reweight(df_valid).values
        else:
            raise ValueError("不支持的重加权器类型。")

        train_pool = Pool(data=x_train, label=y_train_1d, weight=w_train)
        valid_pool = Pool(data=x_valid, label=y_valid_1d, weight=w_valid)

        # 初始化catboost模型
        self._params["iterations"] = num_boost_round
        self._params["early_stopping_rounds"] = early_stopping_rounds
        self._params["verbose_eval"] = verbose_eval
        self._params["task_type"] = "GPU" if get_gpu_device_count() > 0 else "CPU"
        self.model = CatBoost(self._params, **kwargs)

        # 训练模型
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True, **kwargs)

        evals_result = self.model.get_evals_result()
        evals_result["train"] = list(evals_result["learn"].values())[0]
        evals_result["valid"] = list(evals_result["validation"].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("模型尚未训练！")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """获取特征重要性

        注意
        -----
            参数参考：
            https://catboost.ai/docs/concepts/python-reference_catboost_get_feature_importance.html#python-reference_catboost_get_feature_importance
        """
        return pd.Series(
            data=self.model.get_feature_importance(*args, **kwargs), index=self.model.feature_names_
        ).sort_values(ascending=False)


if __name__ == "__main__":
    cat = CatBoostModel()
