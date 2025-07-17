# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Tuple, TypeVar

from qlib.typehint import final

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper

SimulatorState = TypeVar("SimulatorState")


class Reward(Generic[SimulatorState]):
    """
    奖励计算组件，接受单个参数：模拟器状态。返回一个实数：奖励。

    子类应实现``reward(simulator_state)``来实现自定义奖励计算逻辑。
    """

    env: Optional[EnvWrapper] = None

    @final
    def __call__(self, simulator_state: SimulatorState) -> float:
        return self.reward(simulator_state)

    def reward(self, simulator_state: SimulatorState) -> float:
        """实现此方法以定义自定义奖励。"""
        raise NotImplementedError("请在`reward()`中实现奖励计算逻辑。")

    def log(self, name: str, value: Any) -> None:
        """记录奖励相关指标到日志。"""
        assert self.env is not None
        self.env.logger.add_scalar(name, value)


class RewardCombination(Reward):
    """多个奖励的组合。"""

    def __init__(self, rewards: Dict[str, Tuple[Reward, float]]) -> None:
        """初始化奖励组合。

        参数
        ----------
        rewards
            字典，键为奖励名称，值为元组(reward_fn, weight)，其中reward_fn是奖励函数，weight是权重。
        """
        self.rewards = rewards

    def reward(self, simulator_state: Any) -> float:
        """计算组合奖励，将多个奖励加权求和。

        参数
        ----------
        simulator_state
            模拟器状态，用于计算各个奖励。

        返回
        -------
        加权求和后的总奖励。
        """
        total_reward = 0.0
        for name, (reward_fn, weight) in self.rewards.items():
            rew = reward_fn(simulator_state) * weight
            total_reward += rew
            self.log(name, rew)
        return total_reward


# TODO:
# reward_factory目前已禁用

# _RegistryConfigReward = RegistryConfig[REWARDS]


# @configclass
# class _WeightedRewardConfig:
#     weight: float
#     reward: _RegistryConfigReward


# RewardConfig = Union[_RegistryConfigReward, Dict[str, Union[_RegistryConfigReward, _WeightedRewardConfig]]]


# def reward_factory(reward_config: RewardConfig) -> Reward:
#     """
#     使用此工厂从配置实例化奖励。
#     直接使用``reward_config.build()``可能无法工作，因为奖励可能有复杂的组合。
#     """
#     if isinstance(reward_config, dict):
#         # 作为奖励组合
#         rewards = {}
#         for name, rew in reward_config.items():
#             if not isinstance(rew, _WeightedRewardConfig):
#                 # 默认权重为1
#                 rew = _WeightedRewardConfig(weight=1., rew=rew)
#             # 此步骤不递归构建
#             rewards[name] = (rew.reward.build(), rew.weight)
#         return RewardCombination(rewards)
#     else:
#         # 单个奖励
#         return reward_config.build()
