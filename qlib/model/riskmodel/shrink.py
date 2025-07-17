import numpy as np
from typing import Union

from qlib.model.riskmodel import RiskModel


class ShrinkCovEstimator(RiskModel):
    """收缩协方差估计器

    该估计器会将样本协方差矩阵向单位矩阵收缩:
        S_hat = (1 - alpha) * S + alpha * F
    其中`alpha`是收缩参数，`F`是收缩目标。

    支持以下收缩参数(`alpha`):
        - `lw` [1][2][3]: 使用Ledoit-Wolf收缩参数
        - `oas` [4]: 使用Oracle Approximating Shrinkage收缩参数
        - float: 直接指定收缩参数，应在[0, 1]之间

    支持以下收缩目标(`F`):
        - `const_var` [1][4][5]: 假设股票具有相同的常数方差和零相关性
        - `const_corr` [2][6]: 假设股票具有不同的方差但相等的相关性
        - `single_factor` [3][7]: 假设单因子模型作为收缩目标
        - np.ndarray: 直接提供收缩目标

    注意:
        - 最优收缩参数取决于收缩目标的选择。
            目前`oas`不支持`const_corr`和`single_factor`。
        - 如果数据有缺失值，请记得将`nan_option`设置为`fill`或`mask`。

    参考文献:
        [1] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
            Journal of Multivariate Analysis, 88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
        [2] Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix.
            Journal of Portfolio Management, 30(4), 1–22. https://doi.org/10.3905/jpm.2004.110
        [3] Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns
            with an application to portfolio selection.
            Journal of Empirical Finance, 10(5), 603–621. https://doi.org/10.1016/S0927-5398(03)00007-0
        [4] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance
            estimation. IEEE Transactions on Signal Processing, 58(10), 5016–5029.
            https://doi.org/10.1109/TSP.2010.2053029
        [5] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-00007f64e5b9/cov1para.m.zip
        [6] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip
        [7] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-0000648dfc98/covMarket.m.zip
    """

    SHR_LW = "lw"
    SHR_OAS = "oas"

    TGT_CONST_VAR = "const_var"
    TGT_CONST_CORR = "const_corr"
    TGT_SINGLE_FACTOR = "single_factor"

    def __init__(self, alpha: Union[str, float] = 0.0, target: Union[str, np.ndarray] = "const_var", **kwargs):
        """
        参数:
            alpha (str or float): 收缩参数或估计器(`lw`/`oas`)
            target (str or np.ndarray): 收缩目标(`const_var`/`const_corr`/`single_factor`)
            kwargs: 更多参数请参见`RiskModel`
        """
        super().__init__(**kwargs)

        # alpha
        if isinstance(alpha, str):
            assert alpha in [self.SHR_LW, self.SHR_OAS], f"shrinking method `{alpha}` is not supported"
        elif isinstance(alpha, (float, np.floating)):
            assert 0 <= alpha <= 1, "alpha should be between [0, 1]"
        else:
            raise TypeError("invalid argument type for `alpha`")
        self.alpha = alpha

        # target
        if isinstance(target, str):
            assert target in [
                self.TGT_CONST_VAR,
                self.TGT_CONST_CORR,
                self.TGT_SINGLE_FACTOR,
            ], f"shrinking target `{target} is not supported"
        elif isinstance(target, np.ndarray):
            pass
        else:
            raise TypeError("invalid argument type for `target`")
        if alpha == self.SHR_OAS and target != self.TGT_CONST_VAR:
            raise NotImplementedError("currently `oas` can only support `const_var` as target")
        self.target = target

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # sample covariance
        S = super()._predict(X)

        # shrinking target
        F = self._get_shrink_target(X, S)

        # get shrinking parameter
        alpha = self._get_shrink_param(X, S, F)

        # shrink covariance
        if alpha > 0:
            S *= 1 - alpha
            F *= alpha
            S += F

        return S

    def _get_shrink_target(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """获取收缩目标`F`"""
        if self.target == self.TGT_CONST_VAR:
            return self._get_shrink_target_const_var(X, S)
        if self.target == self.TGT_CONST_CORR:
            return self._get_shrink_target_const_corr(X, S)
        if self.target == self.TGT_SINGLE_FACTOR:
            return self._get_shrink_target_single_factor(X, S)
        return self.target

    def _get_shrink_target_const_var(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """获取常数方差收缩目标

        该目标假设零成对相关性和常数方差。
        常数方差通过平均所有样本方差来估计。
        """
        n = len(S)
        F = np.eye(n)
        np.fill_diagonal(F, np.mean(np.diag(S)))
        return F

    def _get_shrink_target_const_corr(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """获取常数相关性收缩目标

        该目标假设常数成对相关性但保持样本方差。
        常数相关性通过平均所有成对相关性来估计。
        """
        n = len(S)
        var = np.diag(S)
        sqrt_var = np.sqrt(var)
        covar = np.outer(sqrt_var, sqrt_var)
        r_bar = (np.sum(S / covar) - n) / (n * (n - 1))
        F = r_bar * covar
        np.fill_diagonal(F, var)
        return F

    def _get_shrink_target_single_factor(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """获取单因子模型收缩目标"""
        X_mkt = np.nanmean(X, axis=1)
        cov_mkt = np.asarray(X.T.dot(X_mkt) / len(X))
        var_mkt = np.asarray(X_mkt.dot(X_mkt) / len(X))
        F = np.outer(cov_mkt, cov_mkt) / var_mkt
        np.fill_diagonal(F, np.diag(S))
        return F

    def _get_shrink_param(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """获取收缩参数`alpha`

        注意:
            Ledoit-Wolf收缩参数估计器包含三种不同方法。
        """
        if self.alpha == self.SHR_OAS:
            return self._get_shrink_param_oas(X, S, F)
        elif self.alpha == self.SHR_LW:
            if self.target == self.TGT_CONST_VAR:
                return self._get_shrink_param_lw_const_var(X, S, F)
            if self.target == self.TGT_CONST_CORR:
                return self._get_shrink_param_lw_const_corr(X, S, F)
            if self.target == self.TGT_SINGLE_FACTOR:
                return self._get_shrink_param_lw_single_factor(X, S, F)
        return self.alpha

    def _get_shrink_param_oas(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Oracle近似收缩估计器

        该方法使用以下公式估计收缩协方差估计器的`alpha`参数:
            A = (1 - 2 / p) * trace(S^2) + trace^2(S)
            B = (n + 1 - 2 / p) * (trace(S^2) - trace^2(S) / p)
            alpha = A / B
        其中`n`和`p`分别是观测值和变量的维度。
        """
        trS2 = np.sum(S**2)
        tr2S = np.trace(S) ** 2

        n, p = X.shape

        A = (1 - 2 / p) * (trS2 + tr2S)
        B = (n + 1 - 2 / p) * (trS2 + tr2S / p)
        alpha = A / B

        return alpha

    def _get_shrink_param_lw_const_var(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Ledoit-Wolf收缩估计器(常数方差)

        该方法将协方差矩阵向常数方差目标收缩。
        """
        t, n = X.shape

        y = X**2
        phi = np.sum(y.T.dot(y) / t - S**2)

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = phi / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha

    def _get_shrink_param_lw_const_corr(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Ledoit-Wolf收缩估计器(常数相关性)

        该方法将协方差矩阵向常数相关性目标收缩。
        """
        t, n = X.shape

        var = np.diag(S)
        sqrt_var = np.sqrt(var)
        r_bar = (np.sum(S / np.outer(sqrt_var, sqrt_var)) - n) / (n * (n - 1))

        y = X**2
        phi_mat = y.T.dot(y) / t - S**2
        phi = np.sum(phi_mat)

        theta_mat = (X**3).T.dot(X) / t - var[:, None] * S
        np.fill_diagonal(theta_mat, 0)
        rho = np.sum(np.diag(phi_mat)) + r_bar * np.sum(np.outer(1 / sqrt_var, sqrt_var) * theta_mat)

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = (phi - rho) / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha

    def _get_shrink_param_lw_single_factor(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Ledoit-Wolf收缩估计器(单因子模型)

        该方法将协方差矩阵向单因子模型目标收缩。
        """
        t, n = X.shape

        X_mkt = np.nanmean(X, axis=1)
        cov_mkt = np.asarray(X.T.dot(X_mkt) / len(X))
        var_mkt = np.asarray(X_mkt.dot(X_mkt) / len(X))

        y = X**2
        phi = np.sum(y.T.dot(y)) / t - np.sum(S**2)

        rdiag = np.sum(y**2) / t - np.sum(np.diag(S) ** 2)
        z = X * X_mkt[:, None]
        v1 = y.T.dot(z) / t - cov_mkt[:, None] * S
        roff1 = np.sum(v1 * cov_mkt[:, None].T) / var_mkt - np.sum(np.diag(v1) * cov_mkt) / var_mkt
        v3 = z.T.dot(z) / t - var_mkt * S
        roff3 = np.sum(v3 * np.outer(cov_mkt, cov_mkt)) / var_mkt**2 - np.sum(np.diag(v3) * cov_mkt**2) / var_mkt**2
        roff = 2 * roff1 - roff3
        rho = rdiag + roff

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = (phi - rho) / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha
