# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module covers some utility functions that operate on data or basic object
"""
from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd

from qlib.data.data import DatasetProvider


def robust_zscore(x: pd.Series, zscore=False):
    """鲁棒ZScore标准化

    使用鲁棒统计量进行Z-Score标准化:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    参考:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    x = x - x.median()
    mad = x.abs().median()
    x = np.clip(x / mad / 1.4826, -3, 3)
    if zscore:
        x -= x.mean()
        x /= x.std()
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std())


def deepcopy_basic_type(obj: object) -> object:
    """
    深度复制对象但不复制复杂对象
        当需要生成Qlib任务并共享处理器时非常有用

    注意:
    - 此函数无法处理递归对象!!!!!

    参数
    ----------
    obj : object
        待复制的对象

    返回
    -------
    object:
        复制后的对象
    """
    if isinstance(obj, tuple):
        return tuple(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, list):
        return list(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: deepcopy_basic_type(v) for k, v in obj.items()}
    else:
        return obj


S_DROP = "__DROP__"  # this is a symbol which indicates drop the value


def update_config(base_config: dict, ext_config: Union[dict, List[dict]]):
    """
    支持基于扩展配置更新基础配置

    >>> bc = {"a": "xixi"}
    >>> ec = {"b": "haha"}
    >>> new_bc = update_config(bc, ec)
    >>> print(new_bc)
    {'a': 'xixi', 'b': 'haha'}
    >>> print(bc)  # 基础配置不应被改变
    {'a': 'xixi'}
    >>> print(update_config(bc, {"b": S_DROP}))
    {'a': 'xixi'}
    >>> print(update_config(new_bc, {"b": S_DROP}))
    {'a': 'xixi'}
    """

    base_config = deepcopy(base_config)  # in case of modifying base config

    for ec in ext_config if isinstance(ext_config, (list, tuple)) else [ext_config]:
        for key in ec:
            if key not in base_config:
                # if it is not in the default key, then replace it.
                # ADD if not drop
                if ec[key] != S_DROP:
                    base_config[key] = ec[key]

            else:
                if isinstance(base_config[key], dict) and isinstance(ec[key], dict):
                    # Recursive
                    # Both of them are dict, then update it nested
                    base_config[key] = update_config(base_config[key], ec[key])
                elif ec[key] == S_DROP:
                    # DROP
                    del base_config[key]
                else:
                    # REPLACE
                    # one of then are not dict. Then replace
                    base_config[key] = ec[key]
    return base_config


def guess_horizon(label: List):
    """
    尝试通过解析标签猜测horizon
    """
    expr = DatasetProvider.parse_fields(label)[0]
    lft_etd, rght_etd = expr.get_extended_window_size()
    return rght_etd
