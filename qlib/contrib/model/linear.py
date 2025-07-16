# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Text, Union
from qlib.log import get_module_logger
from qlib.data.dataset.weight import Reweighter
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class LinearModel(Model):
    """线性模型

    解决以下回归问题之一：
        - `ols`: 最小化 |y - Xw|^2_2
        - `nnls`: 最小化 |y - Xw|^2_2，约束条件 w >= 0
        - `ridge`: 最小化 |y - Xw|^2_2 + \alpha*|w|^2_2
        - `lasso`: 最小化 |y - Xw|^2_2 + \alpha*|w|_1
    其中 `w` 是回归系数。
    """

    OLS = "ols"
    NNLS = "nnls"
    RIDGE = "ridge"
    LASSO = "lasso"

    def __init__(self, estimator="ols", alpha=0.0, fit_intercept=False, include_valid: bool = False):
        """
        参数
        ----------
        estimator : str
            用于线性回归的估计器类型
        alpha : float
            l1或l2正则化参数
        fit_intercept : bool
            是否拟合截距项
        include_valid: bool
            是否将验证集数据包含在训练中？
            应包含验证集数据
        """
        assert estimator in [self.OLS, self.NNLS, self.RIDGE, self.LASSO], f"不支持的估计器 `{estimator}`"
        self.estimator = estimator

        assert alpha == 0 or estimator in [self.RIDGE, self.LASSO], f"alpha仅在`ridge`和`lasso`中支持"
        self.alpha = alpha

        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.include_valid = include_valid

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if self.include_valid:
            try:
                df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                df_train = pd.concat([df_train, df_valid])
            except KeyError:
                get_module_logger("LinearModel").info("include_valid=True，但验证集不存在")
        df_train = df_train.dropna()
        if df_train.empty:
            raise ValueError("数据集数据为空，请检查您的数据集配置。")
        if reweighter is not None:
            w: pd.Series = reweighter.reweight(df_train)
            w = w.values
        else:
            w = None
        X, y = df_train["feature"].values, np.squeeze(df_train["label"].values)

        if self.estimator in [self.OLS, self.RIDGE, self.LASSO]:
            self._fit(X, y, w)
        elif self.estimator == self.NNLS:
            self._fit_nnls(X, y, w)
        else:
            raise ValueError(f"unknown estimator `{self.estimator}`")

        return self

    def _fit(self, X, y, w):
        if self.estimator == self.OLS:
            model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=False)
        else:
            model = {self.RIDGE: Ridge, self.LASSO: Lasso}[self.estimator](
                alpha=self.alpha, fit_intercept=self.fit_intercept, copy_X=False
            )
        model.fit(X, y, sample_weight=w)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def _fit_nnls(self, X, y, w=None):
        if w is not None:
            raise NotImplementedError("TODO: 支持带权重的nnls")  # TODO
        if self.fit_intercept:
            X = np.c_[X, np.ones(len(X))]  # NOTE: mem copy
        coef = nnls(X, y)[0]
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0.0

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.coef_ is None:
            raise ValueError("模型尚未训练！")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(x_test.values @ self.coef_ + self.intercept_, index=x_test.index)
