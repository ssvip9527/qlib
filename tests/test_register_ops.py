# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np

from qlib.data import D
from qlib.data.ops import ElemOperator, PairOperator
from qlib.tests import TestAutoData


class Diff(ElemOperator):
    """特征一阶差分
    参数
    ----------
    feature : Expression
        特征实例
    返回
    ----------
    Expression
        一阶差分后的特征实例
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.diff()

    def get_extended_window_size(self):
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        return lft_etd + 1, rght_etd


class Distance(PairOperator):
    """特征距离
    参数
    ----------
    feature : Expression
        特征实例
    返回
    ----------
    Expression
        距离特征实例
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_left = self.feature_left.load(instrument, start_index, end_index, freq)
        series_right = self.feature_right.load(instrument, start_index, end_index, freq)
        return np.abs(series_left - series_right)


class TestRegiterCustomOps(TestAutoData):
    @classmethod
    def setUpClass(cls) -> None:
        # 更新设置参数，添加自定义算子
        cls._setup_kwargs.update({"custom_ops": [Diff, Distance]})
        super().setUpClass()

    def test_regiter_custom_ops(self):
        # 测试自定义算子注册功能
        instruments = ["SH600000"]
        fields = ["Diff($close)", "Distance($close, Ref($close, 1))"]
        print(D.features(instruments, fields, start_time="2010-01-01", end_time="2017-12-31", freq="day"))


if __name__ == "__main__":
    unittest.main()
