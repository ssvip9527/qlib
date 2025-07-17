# Copyright (c) Microsoft Corporation.
# MIT许可证授权。

"""
Group可以根据`group_func`对一组对象进行分组并将其转换为字典。
分组后，我们提供了一种方法来归约它们。

例如：

group: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}
reduce: {(A,B): {C1: object, C2: object}} -> {(A,B): object}

"""

from qlib.model.ens.ensemble import Ensemble, RollingEnsemble
from typing import Callable
from joblib import Parallel, delayed


class Group:
    """Group the objects based on dict"""

    def __init__(self, group_func=None, ens: Ensemble = None):
        """
        初始化Group。

        参数:
            group_func (Callable, optional): 给定一个字典并返回分组键和其中一个分组元素。

                For example: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}

            默认为None。

            ens (Ensemble, optional): 如果不为None，则在分组后对分组值进行集成。
        """
        self._group_func = group_func
        self._ens_func = ens

    def group(self, *args, **kwargs) -> dict:
        """
        将一组对象分组并转换为字典。

        For example: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}

        返回:
            dict: 分组后的字典
        """
        if isinstance(getattr(self, "_group_func", None), Callable):
            return self._group_func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Please specify valid `group_func`.")

    def reduce(self, *args, **kwargs) -> dict:
        """
        归约分组后的字典。

        For example: {(A,B): {C1: object, C2: object}} -> {(A,B): object}

        返回:
            dict: 归约后的字典
        """
        if isinstance(getattr(self, "_ens_func", None), Callable):
            return self._ens_func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Please specify valid `_ens_func`.")

    def __call__(self, ungrouped_dict: dict, n_jobs: int = 1, verbose: int = 0, *args, **kwargs) -> dict:
        """
        将未分组的字典分成不同的组。

        参数:
            ungrouped_dict (dict): 待分组的字典，格式如 {name: things}

        返回:
            dict: 分组后的字典，格式如 {G1: 对象, G2: 对象}
            n_jobs: 需要的进程数。
            verbose: Parallel的打印模式。
        """

        # NOTE: The multiprocessing will raise error if you use `Serializable`
        # Because the `Serializable` will affect the behaviors of pickle
        grouped_dict = self.group(ungrouped_dict, *args, **kwargs)

        key_l = []
        job_l = []
        for key, value in grouped_dict.items():
            key_l.append(key)
            job_l.append(delayed(Group.reduce)(self, value))
        return dict(zip(key_l, Parallel(n_jobs=n_jobs, verbose=verbose)(job_l)))


class RollingGroup(Group):
    """滚动字典分组"""

    def group(self, rolling_dict: dict) -> dict:
        """给定一个滚动字典如{(A,B,R): things}，返回分组后的字典如{(A,B): {R:things}}

        注意：这里假设滚动键在键元组的末尾，因为滚动结果通常需要先进行集成。

        参数:
            rolling_dict (dict): 滚动字典。如果键不是元组，则不进行任何操作。

        返回:
            dict: 分组后的字典
        """
        grouped_dict = {}
        for key, values in rolling_dict.items():
            if isinstance(key, tuple):
                grouped_dict.setdefault(key[:-1], {})[key[-1]] = values
            else:
                raise TypeError(f"Expected `tuple` type, but got a value `{key}`")
        return grouped_dict

    def __init__(self, ens=RollingEnsemble()):
        super().__init__(ens=ens)
