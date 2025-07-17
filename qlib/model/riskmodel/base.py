# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import numpy as np
import pandas as pd
from typing import Union

from qlib.model.base import BaseModel


class RiskModel(BaseModel):
    """风险模型

    用于估计股票收益的协方差矩阵。
    """

    MASK_NAN = "mask"
    FILL_NAN = "fill"
    IGNORE_NAN = "ignore"

    def __init__(self, nan_option: str = "ignore", assume_centered: bool = False, scale_return: bool = True):
        """
        参数:
            nan_option (str): NaN值处理选项(`ignore`/`mask`/`fill`)。
            assume_centered (bool): 是否假设数据已中心化。
            scale_return (bool): 是否将收益率缩放为百分比。
        """
        # nan
        assert nan_option in [
            self.MASK_NAN,
            self.FILL_NAN,
            self.IGNORE_NAN,
        ], f"`nan_option={nan_option}` is not supported"
        self.nan_option = nan_option

        self.assume_centered = assume_centered
        self.scale_return = scale_return

    def predict(
        self,
        X: Union[pd.Series, pd.DataFrame, np.ndarray],
        return_corr: bool = False,
        is_price: bool = True,
        return_decomposed_components=False,
    ) -> Union[pd.DataFrame, np.ndarray, tuple]:
        """
        参数:
            X (pd.Series, pd.DataFrame or np.ndarray): 用于估计协方差的数据，
                变量为列，观测为行。
            return_corr (bool): 是否返回相关矩阵。
            is_price (bool): `X`是否包含价格数据(否则假设为股票收益)。
            return_decomposed_components (bool): 是否返回协方差矩阵的分解成分。

        返回:
            pd.DataFrame or np.ndarray: 估计的协方差(或相关)矩阵。
        """
        assert (
            not return_corr or not return_decomposed_components
        ), "Can only return either correlation matrix or decomposed components."

        # transform input into 2D array
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            columns = None
        else:
            if isinstance(X.index, pd.MultiIndex):
                if isinstance(X, pd.DataFrame):
                    X = X.iloc[:, 0].unstack(level="instrument")  # always use the first column
                else:
                    X = X.unstack(level="instrument")
            else:
                # X is 2D DataFrame
                pass
            columns = X.columns  # will be used to restore dataframe
            X = X.values

        # calculate pct_change
        if is_price:
            X = X[1:] / X[:-1] - 1  # NOTE: resulting `n - 1` rows

        # scale return
        if self.scale_return:
            X *= 100

        # handle nan and centered
        X = self._preprocess(X)

        # return decomposed components if needed
        if return_decomposed_components:
            assert (
                "return_decomposed_components" in inspect.getfullargspec(self._predict).args
            ), "This risk model does not support return decomposed components of the covariance matrix "

            F, cov_b, var_u = self._predict(X, return_decomposed_components=True)  # pylint: disable=E1123
            return F, cov_b, var_u

        # estimate covariance
        S = self._predict(X)

        # return correlation if needed
        if return_corr:
            vola = np.sqrt(np.diag(S))
            corr = S / np.outer(vola, vola)
            if columns is None:
                return corr
            return pd.DataFrame(corr, index=columns, columns=columns)

        # return covariance
        if columns is None:
            return S
        return pd.DataFrame(S, index=columns, columns=columns)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """协方差估计实现

        此方法应由子类重写。

        默认实现经验协方差估计。

        参数:
            X (np.ndarray): 包含多个变量(列)和观测(行)的数据矩阵。

        返回:
            np.ndarray: 协方差矩阵。
        """
        xTx = np.asarray(X.T.dot(X))
        N = len(X)
        if isinstance(X, np.ma.MaskedArray):
            M = 1 - X.mask
            N = M.T.dot(M)  # each pair has distinct number of samples
        return xTx / N

    def _preprocess(self, X: np.ndarray) -> Union[np.ndarray, np.ma.MaskedArray]:
        """处理NaN值并中心化数据

        注意:
            如果`nan_option='mask'`则返回的数组将是`np.ma.MaskedArray`。
        """
        # handle nan
        if self.nan_option == self.FILL_NAN:
            X = np.nan_to_num(X)
        elif self.nan_option == self.MASK_NAN:
            X = np.ma.masked_invalid(X)
        # centralize
        if not self.assume_centered:
            X = X - np.nanmean(X, axis=0)
        return X
