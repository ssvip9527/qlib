# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, cast

from tianshou.policy import BasePolicy

from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.utils import FiniteEnvType, LogWriter

from .trainer import Trainer
from .vessel import TrainingVessel


def train(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    initial_states: Sequence[InitialStateType],
    policy: BasePolicy,
    reward: Reward,
    vessel_kwargs: Dict[str, Any],
    trainer_kwargs: Dict[str, Any],
) -> None:
    """使用RL框架提供的并行能力训练策略。

    实验性API，参数可能会变更。

    参数
    ----------
    simulator_fn
        接收初始种子并返回模拟器的可调用对象。
    state_interpreter
        解释模拟器状态。
    action_interpreter
        解释策略动作。
    initial_states
        初始状态集合，每个状态将恰好运行一次。
    policy
        待训练的策略。
    reward
        奖励函数。
    vessel_kwargs
        传递给:class:`TrainingVessel`的关键字参数，如``episode_per_iter``。
    trainer_kwargs
        传递给:class:`Trainer`的关键字参数，如``finite_env_type``, ``concurrency``。
    """

    vessel = TrainingVessel(
        simulator_fn=simulator_fn,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        policy=policy,
        train_initial_states=initial_states,
        reward=reward,  # ignore none
        **vessel_kwargs,
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(vessel)


def backtest(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    initial_states: Sequence[InitialStateType],
    policy: BasePolicy,
    logger: LogWriter | List[LogWriter],
    reward: Reward | None = None,
    finite_env_type: FiniteEnvType = "subproc",
    concurrency: int = 2,
) -> None:
    """使用RL框架提供的并行能力进行回测。

    实验性API，参数可能会变更。

    参数
    ----------
    simulator_fn
        接收初始种子并返回模拟器的可调用对象。
    state_interpreter
        解释模拟器状态。
    action_interpreter
        解释策略动作。
    initial_states
        初始状态集合，每个状态将恰好运行一次。
    policy
        待测试的策略。
    logger
        记录回测结果的日志器。必须提供日志器，否则所有信息都将丢失。
    reward
        可选的奖励函数。对于回测，仅用于测试和记录奖励。
    finite_env_type
        有限环境实现类型。
    concurrency
        并行工作线程数。
    """

    vessel = TrainingVessel(
        simulator_fn=simulator_fn,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        policy=policy,
        test_initial_states=initial_states,
        reward=cast(Reward, reward),  # ignore none
    )
    trainer = Trainer(
        finite_env_type=finite_env_type,
        concurrency=concurrency,
        loggers=logger,
    )
    trainer.test(vessel)
