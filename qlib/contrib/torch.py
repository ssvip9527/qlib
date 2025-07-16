# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权
"""
此模块并非Qlib的必要组成部分。
它们只是一些方便使用的工具。
不应将其导入到Qlib的核心部分。
"""
import torch
import numpy as np
import pandas as pd


def data_to_tensor(data, device="cpu", raise_error=False):
    if isinstance(data, torch.Tensor):
        if device == "cpu":
            return data.cpu()
        else:
            return data.to(device)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data_to_tensor(torch.from_numpy(data.values).float(), device)
    elif isinstance(data, np.ndarray):
        return data_to_tensor(torch.from_numpy(data).float(), device)
    elif isinstance(data, (tuple, list)):
        return [data_to_tensor(i, device) for i in data]
    elif isinstance(data, dict):
        return {k: data_to_tensor(v, device) for k, v in data.items()}
    else:
        if raise_error:
            raise ValueError(f"Unsupported data type: {type(data)}.")
        else:
            return data
