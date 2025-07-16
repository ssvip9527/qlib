# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import random
import pandas as pd
from ...data import D
from ..model.base import Model


class ScoreFileModel(Model):
    """
    该模型将加载分数文件，并返回分数文件中存在的日期对应的分数。
    """

    def __init__(self, score_path):
        """
        初始化ScoreFileModel实例
        参数:
            score_path: 分数文件路径
        """
        pred_test = pd.read_csv(score_path, index_col=[0, 1], parse_dates=True, infer_datetime_format=True)
        self.pred = pred_test

    def get_data_with_date(self, date, **kwargs):
        """
        根据指定日期获取数据
        参数:
            date: 日期
            **kwargs: 其他关键字参数
        返回:
            score_series: 包含股票ID和对应分数的Series
        """
        score = self.pred.loc(axis=0)[:, date]  # (stock_id, trade_date) 多级索引, pdate中的分数
        score_series = score.reset_index(level="datetime", drop=True)[
            "score"
        ]  # pd.Series ; 索引:stock_id, 数据: score
        return score_series

    def predict(self, x_test, **kwargs):
        """
        预测方法
        参数:
            x_test: 测试数据
            **kwargs: 其他关键字参数
        返回:
            x_test: 输入的测试数据
        """
        return x_test

    def score(self, x_test, **kwargs):
        """
        评分方法
        参数:
            x_test: 测试数据
            **kwargs: 其他关键字参数
        """
        return

    def fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        """
        拟合方法
        参数:
            x_train: 训练数据
            y_train: 训练标签
            x_valid: 验证数据
            y_valid: 验证标签
            w_train: 训练权重
            w_valid: 验证权重
            **kwargs: 其他关键字参数
        """
        return

    def save(self, fname, **kwargs):
        """
        保存模型
        参数:
            fname: 文件名
            **kwargs: 其他关键字参数
        """
        return
