# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import nn

from .utils import preds_to_weight_with_clamp, SingleMetaBase


class TimeWeightMeta(SingleMetaBase):
    def __init__(self, hist_step_n, clip_weight=None, clip_method="clamp"):
        # clip_method包括"tanh"或"clamp"
        super().__init__(hist_step_n, clip_weight, clip_method)
        self.linear = nn.Linear(hist_step_n, 1)
        self.k = nn.Parameter(torch.Tensor([8.0]))

    def forward(self, time_perf, time_belong=None, return_preds=False):
        hist_step_n = self.linear.in_features
        # 注意：重塑顺序非常重要
        time_perf = time_perf.reshape(hist_step_n, time_perf.shape[0] // hist_step_n, *time_perf.shape[1:])
        time_perf = torch.mean(time_perf, dim=1, keepdim=False)

        preds = []
        for i in range(time_perf.shape[1]):
            preds.append(self.linear(time_perf[:, i]))
        preds = torch.cat(preds)
        preds = preds - torch.mean(preds)  # avoid using future information
        preds = preds * self.k
        if return_preds:
            if time_belong is None:
                return preds
            else:
                return time_belong @ preds
        else:
            weights = preds_to_weight_with_clamp(preds, self.clip_weight, self.clip_method)
            if time_belong is None:
                return weights
            else:
                return time_belong @ weights


class PredNet(nn.Module):
    def __init__(self, step, hist_step_n, clip_weight=None, clip_method="tanh", alpha: float = 0.0):
        """
        Parameters
        ----------
        alpha : float
            子模型的正则化（当将元模型与线性子模型对齐时有用）
        """
        super().__init__()
        self.step = step
        self.twm = TimeWeightMeta(hist_step_n=hist_step_n, clip_weight=clip_weight, clip_method=clip_method)
        self.init_paramters(hist_step_n)
        self.alpha = alpha

    def get_sample_weights(self, X, time_perf, time_belong, ignore_weight=False):
        weights = torch.from_numpy(np.ones(X.shape[0])).float().to(X.device)
        if not ignore_weight:
            if time_perf is not None:
                weights_t = self.twm(time_perf, time_belong)
                weights = weights * weights_t
        return weights

    def forward(self, X, y, time_perf, time_belong, X_test, ignore_weight=False):
        """变量说明请参考MetaTaskDS的文档"""
        weights = self.get_sample_weights(X, time_perf, time_belong, ignore_weight=ignore_weight)
        X_w = X.T * weights.view(1, -1)
        theta = torch.inverse(X_w @ X + self.alpha * torch.eye(X_w.shape[0])) @ X_w @ y
        return X_test @ theta, weights

    def init_paramters(self, hist_step_n):
        self.twm.linear.weight.data = 1.0 / hist_step_n + self.twm.linear.weight.data * 0.01
        self.twm.linear.bias.data.fill_(0.0)
