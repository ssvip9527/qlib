# Copyright (c) Microsoft Corporation.
# 根据MIT许可证授权。

import abc


class BaseOptimizer(abc.ABC):
    """使用优化相关方法构建投资组合"""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> object:
        """生成优化的投资组合配置"""
