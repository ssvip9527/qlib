# Copyright (c) Microsoft Corporation.
# MIT许可证授权。

"""
集成模块可以合并Ensemble中的对象。例如，如果有多个子模型预测，我们可能需要将它们合并为一个集成预测。
"""

from typing import Union
import pandas as pd
from qlib.utils import FLATTEN_TUPLE, flatten_dict
from qlib.log import get_module_logger


class Ensemble:
    """将ensemble_dict合并为一个集成对象。

    例如: {Rollinga_b: 对象, Rollingb_c: 对象} -> 对象

    当调用此类时:

        Args:
            ensemble_dict (dict): the ensemble dict like {name: things} waiting for merging

        Returns:
            object: the ensemble object
    """

    def __call__(self, ensemble_dict: dict, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `__call__` method.")


class SingleKeyEnsemble(Ensemble):
    """
    如果字典中只有一个键值对，则提取该对象使结果更易读。
    {唯一键: 唯一值} -> 唯一值

    如果有超过1个键或少于1个键，则不进行任何操作。
    甚至可以递归运行以使字典更易读。

    注意：默认递归运行。

    当调用此类时：

        参数:
            ensemble_dict (dict): 字典。字典的键将被忽略。

        返回:
            dict: 更易读的字典。
    """

    def __call__(self, ensemble_dict: Union[dict, object], recursion: bool = True) -> object:
        if not isinstance(ensemble_dict, dict):
            return ensemble_dict
        if recursion:
            tmp_dict = {}
            for k, v in ensemble_dict.items():
                tmp_dict[k] = self(v, recursion)
            ensemble_dict = tmp_dict
        keys = list(ensemble_dict.keys())
        if len(keys) == 1:
            ensemble_dict = ensemble_dict[keys[0]]
        return ensemble_dict


class RollingEnsemble(Ensemble):
    """将类似`prediction`或`IC`的滚动数据字典合并为一个集成。

    注意：字典的值必须是pd.DataFrame，并且具有"datetime"索引。

    当调用此类时：

        参数:
            ensemble_dict (dict): 类似{"A": pd.DataFrame, "B": pd.DataFrame}的字典。
            字典的键将被忽略。

        返回:
            pd.DataFrame: 滚动的完整结果。
    """

    def __call__(self, ensemble_dict: dict) -> pd.DataFrame:
        get_module_logger("RollingEnsemble").info(f"keys in group: {list(ensemble_dict.keys())}")
        artifact_list = list(ensemble_dict.values())
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        return artifact


class AverageEnsemble(Ensemble):
    """
    将相同形状的数据字典(如`prediction`或`IC`)进行平均和标准化，合并为一个集成。

    注意：字典的值必须是pd.DataFrame，并且具有"datetime"索引。如果是嵌套字典，则将其展平。

    当调用此类时：

        参数:
            ensemble_dict (dict): 类似{"A": pd.DataFrame, "B": pd.DataFrame}的字典。
            字典的键将被忽略。

        返回:
            pd.DataFrame: 平均和标准化的完整结果。
    """

    def __call__(self, ensemble_dict: dict) -> pd.DataFrame:
        """使用示例:
        from qlib.model.ens.ensemble import AverageEnsemble
        pred_res['new_key_name'] = AverageEnsemble()(predict_dict)

        参数
        ----------
        ensemble_dict : dict
            需要集成的字典

        返回
        -------
        pd.DataFrame
            包含集成结果的字典
        """
        # need to flatten the nested dict
        ensemble_dict = flatten_dict(ensemble_dict, sep=FLATTEN_TUPLE)
        get_module_logger("AverageEnsemble").info(f"keys in group: {list(ensemble_dict.keys())}")
        values = list(ensemble_dict.values())
        # NOTE: this may change the style underlying data!!!!
        # from pd.DataFrame to pd.Series
        results = pd.concat(values, axis=1)
        results = results.groupby("datetime", group_keys=False).apply(lambda df: (df - df.mean()) / df.std())
        results = results.mean(axis=1)
        results = results.sort_index()
        return results
