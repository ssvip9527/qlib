# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from typing import Union
from sklearn.decomposition import PCA, FactorAnalysis

from qlib.model.riskmodel import RiskModel


class StructuredCovEstimator(RiskModel):
    """结构化协方差估计器

    该估计器假设观测值可以通过多个因子预测
        X = B @ F.T + U
    其中`X`包含多个变量(列)的观测值(行)，
    `F`包含所有变量(行)的因子暴露(列)，
    `B`是所有观测值(行)在所有因子(列)上的回归系数矩阵，
    `U`是与`X`形状相同的残差矩阵。

    因此，结构化协方差可以估计为
        cov(X.T) = F @ cov(B.T) @ F.T + diag(var(U))

    在金融领域，主要有三种设计`F`的方法[1][2]:
        - 统计风险模型(SRM): 潜在因子模型主要成分
        - 基本面风险模型(FRM): 人工设计的因子
        - 深度风险模型(DRM): 神经网络设计的因子(类似SRM和DRM的混合)

    在本实现中我们使用潜在因子模型来指定`F`。
    具体支持以下两种潜在因子模型:
        - `pca`: 主成分分析
        - `fa`: 因子分析

    参考文献:
        [1] Fan, J., Liao, Y., & Liu, H. (2016). An overview of the estimation of large covariance and
            precision matrices. Econometrics Journal, 19(1), C1–C32. https://doi.org/10.1111/ectj.12061
        [2] Lin, H., Zhou, D., Liu, W., & Bian, J. (2021). Deep Risk Model: A Deep Learning Solution for
            Mining Latent Risk Factors to Improve Covariance Matrix Estimation. arXiv preprint arXiv:2107.05201.
    """

    FACTOR_MODEL_PCA = "pca"
    FACTOR_MODEL_FA = "fa"
    DEFAULT_NAN_OPTION = "fill"

    def __init__(self, factor_model: str = "pca", num_factors: int = 10, **kwargs):
        """
        参数:
            factor_model (str): 用于估计结构化协方差的潜在因子模型(`pca`/`fa`)。
            num_factors (int): 保留的成分数量。
            kwargs: 更多信息请参考`RiskModel`
        """
        if "nan_option" in kwargs:
            assert kwargs["nan_option"] in [self.DEFAULT_NAN_OPTION], "nan_option={} is not supported".format(
                kwargs["nan_option"]
            )
        else:
            kwargs["nan_option"] = self.DEFAULT_NAN_OPTION

        super().__init__(**kwargs)

        assert factor_model in [
            self.FACTOR_MODEL_PCA,
            self.FACTOR_MODEL_FA,
        ], "factor_model={} is not supported".format(factor_model)
        self.solver = PCA if factor_model == self.FACTOR_MODEL_PCA else FactorAnalysis

        self.num_factors = num_factors

    def _predict(self, X: np.ndarray, return_decomposed_components=False) -> Union[np.ndarray, tuple]:
        """
        协方差估计实现

        参数:
            X (np.ndarray): 包含多个变量(列)和观测值(行)的数据矩阵。
            return_decomposed_components (bool): 是否返回协方差矩阵的分解成分。

        返回:
            tuple 或 np.ndarray: 分解的协方差矩阵或协方差矩阵。
        """

        model = self.solver(self.num_factors, random_state=0).fit(X)

        F = model.components_.T  # variables x factors
        B = model.transform(X)  # observations x factors
        U = X - B @ F.T
        cov_b = np.cov(B.T)  # factors x factors
        var_u = np.var(U, axis=0)  # diagonal

        if return_decomposed_components:
            return F, cov_b, var_u

        cov_x = F @ cov_b @ F.T + np.diag(var_u)

        return cov_x
