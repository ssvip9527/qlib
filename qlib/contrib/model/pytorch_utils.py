# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn


def count_parameters(models_or_parameters, unit="m"):
    """
    此函数用于获取一个（或多个）模型的参数量（以指定存储单位表示）。

    参数
    ----------
    models_or_parameters : PyTorch模型或参数列表。
    unit : 存储大小单位。

    返回值
    -------
    给定模型或参数的参数量。
    """
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    unit = unit.lower()
    if unit in ("kb", "k"):
        counts /= 2**10
    elif unit in ("mb", "m"):
        counts /= 2**20
    elif unit in ("gb", "g"):
        counts /= 2**30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts
