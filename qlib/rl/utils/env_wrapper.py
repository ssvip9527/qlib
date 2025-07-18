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
        # 分配弱引用给包装器
        #
        # 这里使用弱引用的原因：
        # 1. 逻辑上，其他组件应该能够在没有env_wrapper的情况下运行
        #    例如，它们可能在未来存在于strategy_wrapper中
        #    因此注入一个名为"env"的"硬"属性是不合适的
        # 2. 当环境被销毁时，它就会被销毁
        #    我们不希望它默默地存在于某些解释器中
        # 3. 避免循环引用
        # 4. 当组件被序列化时，我们可以无负担地丢弃环境
        #    (虽然这部分尚未实现)
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

            # TODO: 模拟器/观察可能需要种子来预取一些东西
            # 因为只有种子有能力提前完成这项工作

            # 注意：虽然这里重置了日志记录器，但此函数中的日志不会工作
            # 因为我们无法将它们发送到外部
            # 参见 https://github.com/thu-ml/tianshou/issues/605
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

        # 清除上一步的日志信息
        self.logger.reset()

        # 动作是我们从策略中获得的
        self.status["action_history"].append(policy_action)
        action = self.action_interpreter(self.simulator.get_state(), policy_action)

        # 此更新必须在动作解释器之后和模拟器之前进行
        self.status["cur_step"] += 1

        # 使用转换后的动作更新模拟器
        self.simulator.step(action)

        # 首先更新"done"状态，因为reward_fn稍后可能会使用此状态
        done = self.simulator.done()
        self.status["done"] = done

        # 获取状态并计算观察值
        sim_state = self.simulator.get_state()
        obs = self.state_interpreter(sim_state)
        self.status["obs_history"].append(obs)

        # 奖励和额外信息
        if self.reward_fn is not None:
            rew = self.reward_fn(sim_state)
        else:
            # 没有奖励。视为0。
            rew = 0.0
        self.status["reward_history"].append(rew)

        if self.aux_info_collector is not None:
            aux_info = self.aux_info_collector(sim_state)
        else:
            aux_info = {}

        # 最后的日志记录：RL特定的日志
        if done:
            self.logger.add_scalar("steps_per_episode", self.status["cur_step"])
        self.logger.add_scalar("reward", rew)
        self.logger.add_any("obs", obs, loglevel=LogLevel.DEBUG)
        self.logger.add_any("policy_act", policy_action, loglevel=LogLevel.DEBUG)

        info_dict = InfoDict(log=self.logger.logs(), aux_info=aux_info)
        return obs, rew, done, info_dict

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented in EnvWrapper.")
