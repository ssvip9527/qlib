# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qlib.typehint import final
from .simulator import ActType, StateType

ObsType = TypeVar("ObsType")
PolicyActType = TypeVar("PolicyActType")


class Interpreter:
    """解释器是模拟器产生的状态与强化学习策略所需状态之间的媒介。
    解释器是双向的：

    1. 从模拟器状态到策略状态（又称观测），参见:class:`StateInterpreter`。
    2. 从策略动作到模拟器接受的动作，参见:class:`ActionInterpreter`。

    通过继承这两个子类之一来定义自己的解释器。
    此基类仅用于类型检查。

    建议解释器设计为无状态，即在解释器中使用``self.xxx``存储临时信息是反模式。未来可能支持通过调用``self.env.register_state()``注册解释器相关状态，但第一版暂不支持。
    """


class StateInterpreter(Generic[StateType, ObsType], Interpreter):
    """状态解释器，将qlib执行器的执行结果解释为强化学习环境状态"""

    @property
    def observation_space(self) -> gym.Space:
        raise NotImplementedError()

    @final  # no overridden
    def __call__(self, simulator_state: StateType) -> ObsType:
        obs = self.interpret(simulator_state)
        self.validate(obs)
        return obs

    def validate(self, obs: ObsType) -> None:
        """Validate whether an observation belongs to the pre-defined observation space."""
        _gym_space_contains(self.observation_space, obs)

    def interpret(self, simulator_state: StateType) -> ObsType:
        """解释模拟器的状态。

        参数
        ----------
        simulator_state
            通过``simulator.get_state()``获取的模拟器状态。

        返回
        -------
        策略所需的状态，应符合``observation_space``中定义的状态空间。
        """
        raise NotImplementedError("interpret方法未实现!")


class ActionInterpreter(Generic[StateType, PolicyActType, ActType], Interpreter):
    """动作解释器，将强化学习智能体的动作解释为qlib订单"""

    @property
    def action_space(self) -> gym.Space:
        raise NotImplementedError()

    @final  # no overridden
    def __call__(self, simulator_state: StateType, action: PolicyActType) -> ActType:
        self.validate(action)
        obs = self.interpret(simulator_state, action)
        return obs

    def validate(self, action: PolicyActType) -> None:
        """验证动作是否属于预定义的动作空间。"""
        _gym_space_contains(self.action_space, action)

    def interpret(self, simulator_state: StateType, action: PolicyActType) -> ActType:
        """将策略动作转换为模拟器动作。

        参数
        ----------
        simulator_state
            通过``simulator.get_state()``获取的模拟器状态。
        action
            策略给出的原始动作。

        返回
        -------
        模拟器所需的动作。
        """
        raise NotImplementedError("interpret方法未实现!")


def _gym_space_contains(space: gym.Space, x: Any) -> None:
    """gym.Space.contains的增强版本。
    提供更多关于验证失败原因的诊断信息。

    抛出异常而非返回true或false。
    """
    if isinstance(space, spaces.Dict):
        if not isinstance(x, dict) or len(x) != len(space):
            raise GymSpaceValidationError("Sample must be a dict with same length as space.", space, x)
        for k, subspace in space.spaces.items():
            if k not in x:
                raise GymSpaceValidationError(f"Key {k} not found in sample.", space, x)
            try:
                _gym_space_contains(subspace, x[k])
            except GymSpaceValidationError as e:
                raise GymSpaceValidationError(f"Subspace of key {k} validation error.", space, x) from e

    elif isinstance(space, spaces.Tuple):
        if isinstance(x, (list, np.ndarray)):
            x = tuple(x)  # Promote list and ndarray to tuple for contains check
        if not isinstance(x, tuple) or len(x) != len(space):
            raise GymSpaceValidationError("Sample must be a tuple with same length as space.", space, x)
        for i, (subspace, part) in enumerate(zip(space, x)):
            try:
                _gym_space_contains(subspace, part)
            except GymSpaceValidationError as e:
                raise GymSpaceValidationError(f"Subspace of index {i} validation error.", space, x) from e

    else:
        if not space.contains(x):
            raise GymSpaceValidationError("Validation error reported by gym.", space, x)


class GymSpaceValidationError(Exception):
    """Gym空间验证异常，当观测值或动作不符合预定义空间时抛出。"""
    def __init__(self, message: str, space: gym.Space, x: Any) -> None:
        self.message = message
        self.space = space
        self.x = x

    def __str__(self) -> str:
        return f"{self.message}\n  空间: {self.space}\n  样本: {self.x}"
