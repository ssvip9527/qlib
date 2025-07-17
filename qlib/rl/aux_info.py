# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from qlib.typehint import final

from .simulator import StateType

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper


__all__ = ["AuxiliaryInfoCollector"]

AuxInfoType = TypeVar("AuxInfoType")


class AuxiliaryInfoCollector(Generic[StateType, AuxInfoType]):
    """重写此类以从环境中收集自定义辅助信息。"""

    env: Optional[EnvWrapper] = None

    @final
    def __call__(self, simulator_state: StateType) -> AuxInfoType:
        return self.collect(simulator_state)

    def collect(self, simulator_state: StateType) -> AuxInfoType:
        """重写此方法以获取自定义辅助信息。
        通常在多智能体强化学习中有用。

        参数
        ----------
        simulator_state
            通过``simulator.get_state()``获取的模拟器状态。

        返回
        -------
        辅助信息。
        """
        raise NotImplementedError("collect方法未实现!")
