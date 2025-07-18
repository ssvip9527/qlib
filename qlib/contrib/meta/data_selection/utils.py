# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import nn

from qlib.constant import EPS
from qlib.log import get_module_logger


class ICLoss(nn.Module):
    def __init__(self, skip_size=50):
        super().__init__()
        self.skip_size = skip_size

    def forward(self, pred, y, idx):
        """前向传播。
        FIXME:
        - 有时结果与`pandas.corr()`略有不同
        - 这可能是由于模型的精度问题导致的；

        :param pred: 预测值
        :param y: 标签值
        :param idx: 假设索引级别为(date, inst)，且已排序
        """
        prev = None
        diff_point = []
        for i, (date, inst) in enumerate(idx):
            if date != prev:
                diff_point.append(i)
            prev = date
        diff_point.append(None)
        # The lengths of diff_point will be one more larger then diff_point

        ic_all = 0.0
        skip_n = 0
        for start_i, end_i in zip(diff_point, diff_point[1:]):
            pred_focus = pred[start_i:end_i]  # TODO: just for fake
            if pred_focus.shape[0] < self.skip_size:
                # skip some days which have very small amount of stock.
                skip_n += 1
                continue
            y_focus = y[start_i:end_i]
            if pred_focus.std() < EPS or y_focus.std() < EPS:
                # These cases often happend at the end of test data.
                # Usually caused by fillna(0.)
                skip_n += 1
                continue

            ic_day = torch.dot(
                (pred_focus - pred_focus.mean()) / np.sqrt(pred_focus.shape[0]) / pred_focus.std(),
                (y_focus - y_focus.mean()) / np.sqrt(y_focus.shape[0]) / y_focus.std(),
            )
            ic_all += ic_day
        if len(diff_point) - 1 - skip_n <= 0:
            __import__("ipdb").set_trace()
            raise ValueError("No enough data for calculating IC")
        if skip_n > 0:
            get_module_logger("ICLoss").info(
                f"{skip_n}天因标准差为零或有效样本量过小而被跳过。"
            )
        ic_mean = ic_all / (len(diff_point) - 1 - skip_n)
        return -ic_mean  # ic loss


def preds_to_weight_with_clamp(preds, clip_weight=None, clip_method="tanh"):
    """
    裁剪权重。

    参数
    ----------
    clip_weight: float
        裁剪阈值。
    clip_method: str
        裁剪方法。当前可用："clamp"、"tanh"和"sigmoid"。
    """
    if clip_weight is not None:
        if clip_method == "clamp":
            weights = torch.exp(preds)
            weights = weights.clamp(1.0 / clip_weight, clip_weight)
        elif clip_method == "tanh":
            weights = torch.exp(torch.tanh(preds) * np.log(clip_weight))
        elif clip_method == "sigmoid":
            # intuitively assume its sum is 1
            if clip_weight == 0.0:
                weights = torch.ones_like(preds)
            else:
                sm = nn.Sigmoid()
                weights = sm(preds) * clip_weight  # TODO: 此处clip_weight无用。
                weights = weights / torch.sum(weights) * weights.numel()
        else:
            raise ValueError("Unknown clip_method")
    else:
        weights = torch.exp(preds)
    return weights


class SingleMetaBase(nn.Module):
    def __init__(self, hist_n, clip_weight=None, clip_method="clamp"):
        # 方法可以是tanh或clamp
        super().__init__()
        self.clip_weight = clip_weight
        if clip_method in ["tanh", "clamp"]:
            if self.clip_weight is not None and self.clip_weight < 1.0:
                self.clip_weight = 1 / self.clip_weight
        self.clip_method = clip_method

    def is_enabled(self):
        if self.clip_weight is None:
            return True
        if self.clip_method == "sigmoid":
            if self.clip_weight > 0.0:
                return True
        else:
            if self.clip_weight > 1.0:
                return True
        return False
