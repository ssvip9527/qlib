# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter


class XGBModel(Model, FeatureInt):
    """XGBoost模型"""

    def __init__(self, **kwargs):
        self._params = {}
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
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # XGBoost需要一维数组作为标签
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("XGBoost doesn't support multi-label training")

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train)
            w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        dtrain = xgb.DMatrix(x_train.values, label=y_train_1d, weight=w_train)
        dvalid = xgb.DMatrix(x_valid.values, label=y_valid_1d, weight=w_valid)
        self.model = xgb.train(
            self._params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(xgb.DMatrix(x_test)), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """获取特征重要性

        注意
        -------
            参数参考:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)
