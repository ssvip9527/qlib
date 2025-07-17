# 版权所有 (c) 微软公司。
# MIT许可证授权。

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, OrderedDict, Tuple, cast

import gym
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy, PPOPolicy, DQNPolicy

from qlib.rl.trainer.trainer import Trainer

__all__ = ["AllOne", "PPO", "DQN"]


# baselines #


class NonLearnablePolicy(BasePolicy):
    """Tianshou的BasePolicy，带有空的``learn``和``process_fn``方法。

    未来可能会移出此类。
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> Batch:
        return Batch({})


class AllOne(NonLearnablePolicy):
    """前向传播返回全为1的批次。

    在实现某些基线(如TWAP)时很有用。
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, fill_value: float | int = 1.0) -> None:
        super().__init__(obs_space, action_space)

        self.fill_value = fill_value

    def forward(
        self,
        batch: Batch,
        state: dict | Batch | np.ndarray = None,
        **kwargs: Any,
    ) -> Batch:
        return Batch(act=np.full(len(batch), self.fill_value), state=state)


# ppo #


class PPOActor(nn.Module):
    def __init__(self, extractor: nn.Module, action_dim: int) -> None:
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(nn.Linear(cast(int, extractor.output_dim), action_dim), nn.Softmax(dim=-1))

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        info: dict = {},
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feature = self.extractor(to_torch(obs, device=auto_device(self)))
        out = self.layer_out(feature)
        return out, state


class PPOCritic(nn.Module):
    def __init__(self, extractor: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(cast(int, extractor.output_dim), 1)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        info: dict = {},
    ) -> torch.Tensor:
        feature = self.extractor(to_torch(obs, device=auto_device(self)))
        return self.value_out(feature).squeeze(dim=-1)


class PPO(PPOPolicy):
    """tianshou PPOPolicy的包装器。

    区别：

    - 自动创建actor和critic网络。仅支持离散动作空间。
    - 去除actor网络和critic网络之间的重复参数
      (不确定最新版tianshou是否已包含此功能)。
    - 支持加载检查点的``weight_file``参数。
    - 某些参数的默认值与原始版本不同。
    """

    def __init__(
        self,
        network: nn.Module,
        obs_space: gym.Space,
        action_space: gym.Space,
        lr: float,
        weight_decay: float = 0.0,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 1.0,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
    ) -> None:
        assert isinstance(action_space, Discrete)
        actor = PPOActor(network, action_space.n)
        critic = PPOCritic(network)
        optimizer = torch.optim.Adam(
            chain_dedup(actor.parameters(), critic.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        super().__init__(
            actor,
            critic,
            optimizer,
            torch.distributions.Categorical,
            discount_factor=discount_factor,
            max_grad_norm=max_grad_norm,
            reward_normalization=reward_normalization,
            eps_clip=eps_clip,
            value_clip=value_clip,
            vf_coef=vf_coef,
            gae_lambda=gae_lambda,
            max_batchsize=max_batch_size,
            deterministic_eval=deterministic_eval,
            observation_space=obs_space,
            action_space=action_space,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))


DQNModel = PPOActor  # Reuse PPOActor.


class DQN(DQNPolicy):
    """tianshou DQNPolicy的包装器。

    区别：

    - 自动创建模型网络。仅支持离散动作空间。
    - 支持加载检查点的``weight_file``参数。
    """

    def __init__(
        self,
        network: nn.Module,
        obs_space: gym.Space,
        action_space: gym.Space,
        lr: float,
        weight_decay: float = 0.0,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        weight_file: Optional[Path] = None,
    ) -> None:
        assert isinstance(action_space, Discrete)

        model = DQNModel(network, action_space.n)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        super().__init__(
            model,
            optimizer,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))


# 实用工具：这些应该放在单独的(公共)文件中 #


def auto_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")  # fallback to cpu


def set_weight(policy: nn.Module, loaded_weight: OrderedDict) -> None:
    try:
        policy.load_state_dict(loaded_weight)
    except RuntimeError:
        # try again by loading the converted weight
        # https://github.com/thu-ml/tianshou/issues/468
        for k in list(loaded_weight):
            loaded_weight["_actor_critic." + k] = loaded_weight[k]
        policy.load_state_dict(loaded_weight)


def chain_dedup(*iterables: Iterable) -> Generator[Any, None, None]:
    seen = set()
    for iterable in iterables:
        for i in iterable:
            if i not in seen:
                seen.add(i)
                yield i
