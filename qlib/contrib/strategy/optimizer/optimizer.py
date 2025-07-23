# Copyright (c) Microsoft Corporation.
# 根据MIT许可证授权。


import warnings
import numpy as np
import pandas as pd
import scipy.optimize as so
from typing import Optional, Union, Callable, List

from .base import BaseOptimizer


class PortfolioOptimizer(BaseOptimizer):
    """投资组合优化器

    以下优化算法受支持：
        - `gmv`: 全局最小方差投资组合
        - `mvo`: 均值方差优化投资组合
        - `rp`: 风险平价
        - `inv`: 逆波动率

    注意：
        此优化器始终假设全额投资且不允许卖空。
    """

    OPT_GMV = "gmv"
    OPT_MVO = "mvo"
    OPT_RP = "rp"
    OPT_INV = "inv"

    def __init__(
        self,
        method: str = "inv",
        lamb: float = 0,
        delta: float = 0,
        alpha: float = 0.0,
        scale_return: bool = True,
        tol: float = 1e-8,
    ):
        """
        参数:
            method (str): 投资组合优化方法
            lamb (float): 风险厌恶参数（lamb越大表示越关注收益）
            delta (float): 换手率限制
            alpha (float): L2范数正则化系数
            scale_return (bool): 是否缩放收益以匹配协方差矩阵的波动率
            tol (float): 优化终止的容差
        """
        assert method in [self.OPT_GMV, self.OPT_MVO, self.OPT_RP, self.OPT_INV], f"method `{method}` is not supported"
        self.method = method

        assert lamb >= 0, f"risk aversion parameter `lamb` should be positive"
        self.lamb = lamb

        assert delta >= 0, f"turnover limit `delta` should be positive"
        self.delta = delta

        assert alpha >= 0, f"l2 norm regularizer `alpha` should be positive"
        self.alpha = alpha

        self.tol = tol
        self.scale_return = scale_return

    def __call__(
        self,
        S: Union[np.ndarray, pd.DataFrame],
        r: Optional[Union[np.ndarray, pd.Series]] = None,
        w0: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Union[np.ndarray, pd.Series]:
        """
        参数:
            S (np.ndarray or pd.DataFrame): 协方差矩阵
            r (np.ndarray or pd.Series): 预期收益
            w0 (np.ndarray or pd.Series): 初始权重（用于换手率控制）

        返回:
            np.ndarray or pd.Series: 优化后的投资组合配置
        """
        # transform dataframe into array
        index = None
        if isinstance(S, pd.DataFrame):
            index = S.index
            S = S.values

        # transform return
        if r is not None:
            assert len(r) == len(S), "`r` has mismatched shape"
            if isinstance(r, pd.Series):
                assert r.index.equals(index), "`r` has mismatched index"
                r = r.values

        # transform initial weights
        if w0 is not None:
            assert len(w0) == len(S), "`w0` has mismatched shape"
            if isinstance(w0, pd.Series):
                assert w0.index.equals(index), "`w0` has mismatched index"
                w0 = w0.values

        # scale return to match volatility
        if r is not None and self.scale_return:
            r = r / r.std()
            r *= np.sqrt(np.mean(np.diag(S)))

        # optimize
        w = self._optimize(S, r, w0)

        # restore index if needed
        if index is not None:
            w = pd.Series(w, index=index)

        return w

    def _optimize(self, S: np.ndarray, r: Optional[np.ndarray] = None, w0: Optional[np.ndarray] = None) -> np.ndarray:
        # inverse volatility
        if self.method == self.OPT_INV:
            if r is not None:
                warnings.warn("`r` is set but will not be used for `inv` portfolio")
            if w0 is not None:
                warnings.warn("`w0` is set but will not be used for `inv` portfolio")
            return self._optimize_inv(S)

        # global minimum variance
        if self.method == self.OPT_GMV:
            if r is not None:
                warnings.warn("`r` is set but will not be used for `gmv` portfolio")
            return self._optimize_gmv(S, w0)

        # mean-variance
        if self.method == self.OPT_MVO:
            return self._optimize_mvo(S, r, w0)

        # risk parity
        if self.method == self.OPT_RP:
            if r is not None:
                warnings.warn("`r` is set but will not be used for `rp` portfolio")
            return self._optimize_rp(S, w0)

    def _optimize_inv(self, S: np.ndarray) -> np.ndarray:
        """Inverse volatility"""
        vola = np.diag(S) ** 0.5
        w = 1 / vola
        w /= w.sum()
        return w

    def _optimize_gmv(self, S: np.ndarray, w0: Optional[np.ndarray] = None) -> np.ndarray:
        """optimize global minimum variance portfolio

        此方法求解以下优化问题
            min_w w' S w
            s.t. w >= 0, sum(w) == 1
        其中 `S` 是协方差矩阵。
        """
        return self._solve(len(S), self._get_objective_gmv(S), *self._get_constrains(w0))

    def _optimize_mvo(
        self, S: np.ndarray, r: Optional[np.ndarray] = None, w0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """optimize mean-variance portfolio

        此方法求解以下优化问题
            min_w   - w' r + lamb * w' S w
            s.t.   w >= 0, sum(w) == 1
        其中 `S` 是协方差矩阵，`u` 是预期收益，
        `lamb` 是风险厌恶参数。
        """
        return self._solve(len(S), self._get_objective_mvo(S, r), *self._get_constrains(w0))

    def _optimize_rp(self, S: np.ndarray, w0: Optional[np.ndarray] = None) -> np.ndarray:
        """optimize risk parity portfolio

        此方法求解以下优化问题
            min_w sum_i [w_i - (w' S w) / ((S w)_i * N)]**2
            s.t. w >= 0, sum(w) == 1
        其中 `S` 是协方差矩阵，`N` 是股票数量。
        """
        return self._solve(len(S), self._get_objective_rp(S), *self._get_constrains(w0))

    def _get_objective_gmv(self, S: np.ndarray) -> Callable:
        """global minimum variance optimization objective

        优化目标
            min_w w' S w
        """

        def func(x):
            return x @ S @ x

        return func

    def _get_objective_mvo(self, S: np.ndarray, r: np.ndarray = None) -> Callable:
        """mean-variance optimization objective

        优化目标
            min_w - w' r + lamb * w' S w
        """

        def func(x):
            risk = x @ S @ x
            ret = x @ r
            return -ret + self.lamb * risk

        return func

    def _get_objective_rp(self, S: np.ndarray) -> Callable:
        """risk-parity optimization objective

        优化目标
            min_w sum_i [w_i - (w' S w) / ((S w)_i * N)]**2
        """

        def func(x):
            N = len(x)
            Sx = S @ x
            xSx = x @ Sx
            return np.sum((x - xSx / Sx / N) ** 2)

        return func

    def _get_constrains(self, w0: Optional[np.ndarray] = None):
        """optimization constraints

        定义以下约束条件:
            - 禁止卖空和杠杆: 0 <= w <= 1
            - 全额投资: sum(w) == 1
            - 换手率约束: |w - w0| <= delta
        """

        # no shorting and leverage
        bounds = so.Bounds(0.0, 1.0)

        # full investment constraint
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # == 0

        # turnover constraint
        if w0 is not None:
            cons.append({"type": "ineq", "fun": lambda x: self.delta - np.sum(np.abs(x - w0))})  # >= 0

        return bounds, cons

    def _solve(self, n: int, obj: Callable, bounds: so.Bounds, cons: List) -> np.ndarray:
        """求解优化问题

        参数:
            n (int): 参数数量
            obj (callable): 优化目标
            bounds (Bounds): 参数边界
            cons (list): 优化约束
        """
        # add l2 regularization
        wrapped_obj = obj
        if self.alpha > 0:

            def opt_obj(x):
                return obj(x) + self.alpha * np.sum(np.square(x))

            wrapped_obj = opt_obj

        # solve
        x0 = np.ones(n) / n  # init results
        sol = so.minimize(wrapped_obj, x0, bounds=bounds, constraints=cons, tol=self.tol)
        if not sol.success:
            warnings.warn(f"optimization not success ({sol.status})")

        return sol.x
