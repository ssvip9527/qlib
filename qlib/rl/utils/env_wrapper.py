# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import Any, Callable, cast, Dict, Generic, Iterable, Iterator, Optional, Tuple

import gym
from gym import Space

from qlib.rl.aux_info import AuxiliaryInfoCollector
from qlib.rl.interpreter import ActionInterpreter, ObsType, PolicyActType, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import ActType, InitialStateType, Simulator, StateType
from qlib.typehint import TypedDict
from .finite_env import generate_nan_observation
from .log import LogCollector, LogLevel

__all__ = ["InfoDict", "EnvWrapperStatus", "EnvWrapper"]

# in this case, there won't be any seed for simulator
SEED_INTERATOR_MISSING = "_missing_"


class InfoDict(TypedDict):
    """用于``env.step()``第四个返回值的字典类型。"""

    aux_info: dict
    """依赖于辅助信息收集器的任何信息。"""
    log: Dict[str, Any]
    """由LogCollector收集的信息。"""


class EnvWrapperStatus(TypedDict):
    """
    EnvWrapper中使用的状态数据结构。
    这里的字段采用RL语义。
    例如，``obs``表示输入策略的观测值，
    ``action``表示策略返回的原始动作。
    """

    cur_step: int
    done: bool
    initial_state: Optional[Any]
    obs_history: list
    action_history: list
    reward_history: list


class EnvWrapper(
    gym.Env[ObsType, PolicyActType],
    Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType],
):
    """基于Qlib的RL环境，继承自``gym.Env``。
    组件包装器，包含模拟器、状态解释器、动作解释器和奖励函数。

    这是RL训练中模拟器-解释器-策略框架的体现。
    除策略外的所有组件需要组装成一个称为"环境"的对象。
    "环境"被复制到多个工作进程，在tianshou实现中，
    单个策略(agent)与一批环境交互。

    参数
    ----------
    simulator_fn
        模拟器工厂函数。
        当``seed_iterator``存在时，工厂函数接受一个参数(种子/初始状态)，
        否则不接受参数。
    state_interpreter
        状态-观测转换器。
    action_interpreter
        策略-模拟器动作转换器。
    seed_iterator
        种子迭代器。借助:class:`qlib.rl.utils.DataQueue`，
        不同进程的环境工作器可以共享一个``seed_iterator``。
    reward_fn
        接受StateType并返回浮点数的可调用对象(至少单智能体情况下)。
    aux_info_collector
        收集辅助信息，在MARL中可能有用。
    logger
        日志收集器，收集的日志通过``env.step()``返回值传回主进程。

    属性
    ----------
    status : EnvWrapperStatus
        状态指示器，所有术语采用*RL语言*。
        当用户关心RL侧数据时可以使用。
        没有轨迹时可能为None。
    """

    simulator: Simulator[InitialStateType, StateType, ActType]
    seed_iterator: str | Iterator[InitialStateType] | None

    def __init__(
        self,
        simulator_fn: Callable[..., Simulator[InitialStateType, StateType, ActType]],
        state_interpreter: StateInterpreter[StateType, ObsType],
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        seed_iterator: Optional[Iterable[InitialStateType]],
        reward_fn: Reward | None = None,
        aux_info_collector: AuxiliaryInfoCollector[StateType, Any] | None = None,
        logger: LogCollector | None = None,
    ) -> None:
        # Assign weak reference to wrapper.
        #
        # Use weak reference here, because:
        # 1. Logically, the other components should be able to live without an env_wrapper.
        #    For example, they might live in a strategy_wrapper in future.
        #    Therefore injecting a "hard" attribute called "env" is not appropripate.
        # 2. When the environment gets destroyed, it gets destoryed.
        #    We don't want it to silently live inside some interpreters.
        # 3. Avoid circular reference.
        # 4. When the components get serialized, we can throw away the env without any burden.
        #    (though this part is not implemented yet)
        for obj in [state_interpreter, action_interpreter, reward_fn, aux_info_collector]:
            if obj is not None:
                obj.env = weakref.proxy(self)  # type: ignore

        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter

        if seed_iterator is None:
            # In this case, there won't be any seed for simulator
            # We can't set it to None because None actually means something else.
            # If `seed_iterator` is None, it means that it's exhausted.
            self.seed_iterator = SEED_INTERATOR_MISSING
        else:
            self.seed_iterator = iter(seed_iterator)
        self.reward_fn = reward_fn

        self.aux_info_collector = aux_info_collector
        self.logger: LogCollector = logger or LogCollector()
        self.status: EnvWrapperStatus = cast(EnvWrapperStatus, None)

    @property
    def action_space(self) -> Space:
        return self.action_interpreter.action_space

    @property
    def observation_space(self) -> Space:
        return self.state_interpreter.observation_space

    def reset(self, **kwargs: Any) -> ObsType:
        """
        尝试从状态队列获取状态，并用此状态初始化模拟器。
        如果队列耗尽，则生成无效(nan)观测值。
        """

        try:
            if self.seed_iterator is None:
                raise RuntimeError("You can trying to get a state from a dead environment wrapper.")

            # TODO: simulator/observation might need seed to prefetch something
            # as only seed has the ability to do the work beforehands

            # NOTE: though logger is reset here, logs in this function won't work,
            # because we can't send them outside.
            # See https://github.com/thu-ml/tianshou/issues/605
            self.logger.reset()

            if self.seed_iterator is SEED_INTERATOR_MISSING:
                # no initial state
                initial_state = None
                self.simulator = cast(Callable[[], Simulator], self.simulator_fn)()
            else:
                initial_state = next(cast(Iterator[InitialStateType], self.seed_iterator))
                self.simulator = self.simulator_fn(initial_state)

            self.status = EnvWrapperStatus(
                cur_step=0,
                done=False,
                initial_state=initial_state,
                obs_history=[],
                action_history=[],
                reward_history=[],
            )

            self.simulator.env = cast(EnvWrapper, weakref.proxy(self))

            sim_state = self.simulator.get_state()
            obs = self.state_interpreter(sim_state)

            self.status["obs_history"].append(obs)

            return obs

        except StopIteration:
            # The environment should be recycled because it's in a dead state.
            self.seed_iterator = None
            return generate_nan_observation(self.observation_space)

    def step(self, policy_action: PolicyActType, **kwargs: Any) -> Tuple[ObsType, float, bool, InfoDict]:
        """环境步骤。

        结合代码和注释查看此处发生的事件序列。
        """

        if self.seed_iterator is None:
            raise RuntimeError("State queue is already exhausted, but the environment is still receiving action.")

        # Clear the logged information from last step
        self.logger.reset()

        # Action is what we have got from policy
        self.status["action_history"].append(policy_action)
        action = self.action_interpreter(self.simulator.get_state(), policy_action)

        # This update must be after action interpreter and before simulator.
        self.status["cur_step"] += 1

        # Use the converted action of update the simulator
        self.simulator.step(action)

        # Update "done" first, as this status might be used by reward_fn later
        done = self.simulator.done()
        self.status["done"] = done

        # Get state and calculate observation
        sim_state = self.simulator.get_state()
        obs = self.state_interpreter(sim_state)
        self.status["obs_history"].append(obs)

        # Reward and extra info
        if self.reward_fn is not None:
            rew = self.reward_fn(sim_state)
        else:
            # No reward. Treated as 0.
            rew = 0.0
        self.status["reward_history"].append(rew)

        if self.aux_info_collector is not None:
            aux_info = self.aux_info_collector(sim_state)
        else:
            aux_info = {}

        # Final logging stuff: RL-specific logs
        if done:
            self.logger.add_scalar("steps_per_episode", self.status["cur_step"])
        self.logger.add_scalar("reward", rew)
        self.logger.add_any("obs", obs, loglevel=LogLevel.DEBUG)
        self.logger.add_any("policy_act", policy_action, loglevel=LogLevel.DEBUG)

        info_dict = InfoDict(log=self.logger.logs(), aux_info=aux_info)
        return obs, rew, done, info_dict

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented in EnvWrapper.")
