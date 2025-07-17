#  Copyright (c) Microsoft Corporation.
# MIT许可证授权。

"""
模型解释接口
"""

import pandas as pd
from abc import abstractmethod


class FeatureInt:
    """特征解释器"""

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性

        返回
        -------
            索引是特征名称。
            值越大，重要性越高。
        """


class LightGBMFInt(FeatureInt):
    """LightGBM特征解释器"""

    def __init__(self):
        self.model = None

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """获取特征重要性

        说明
        -----
            参数参考:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        return pd.Series(
            self.model.feature_importance(*args, **kwargs), index=self.model.feature_name()
        ).sort_values(  # pylint: disable=E1101
            ascending=False
        )
