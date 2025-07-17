# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, Generic, Iterable, Sequence, TypeVar, cast

import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy

from qlib.constant import INF
from qlib.log import get_module_logger
from qlib.rl.interpreter import ActionInterpreter, ActType, ObsType, PolicyActType, StateInterpreter, StateType
from qlib.rl.reward import Reward
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.utils import DataQueue
from qlib.rl.utils.finite_env import FiniteVectorEnv

if TYPE_CHECKING:
    from .trainer import Trainer


T = TypeVar("T")
_logger = get_module_logger(__name__)


class SeedIteratorNotAvailable(BaseException):
    pass


class TrainingVesselBase(Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType]):
    """包含模拟器、解释器和策略的容器，将被发送给训练器。
    此类控制训练中与算法相关的部分，而训练器负责运行时部分。

    该容器还定义了核心训练部分最重要的逻辑，
    以及(可选)一些回调函数用于在特定事件插入自定义逻辑。
    """

    simulator_fn: Callable[[InitialStateType], Simulator[InitialStateType, StateType, ActType]]
    state_interpreter: StateInterpreter[StateType, ObsType]
    action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType]
    policy: BasePolicy
    reward: Reward
    trainer: Trainer

    def assign_trainer(self, trainer: Trainer) -> None:
        self.trainer = weakref.proxy(trainer)  # type: ignore

    def train_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        """重写此方法以创建训练用的种子迭代器。
        如果可迭代对象是上下文管理器，整个训练将在with块中调用，
        并且迭代器会在训练完成后自动关闭。"""
        raise SeedIteratorNotAvailable("Seed iterator for training is not available.")

    def val_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        """重写此方法以创建验证用的种子迭代器。"""
        raise SeedIteratorNotAvailable("Seed iterator for validation is not available.")

    def test_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        """重写此方法以创建测试用的种子迭代器。"""
        raise SeedIteratorNotAvailable("Seed iterator for testing is not available.")

    def train(self, vector_env: BaseVectorEnv) -> Dict[str, Any]:
        """实现此方法以进行一次训练迭代。在RL中，一次迭代通常指一次收集。"""
        raise NotImplementedError()

    def validate(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        """实现此方法以对策略进行一次验证。"""
        raise NotImplementedError()

    def test(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        """实现此方法以在测试环境中评估策略一次。"""
        raise NotImplementedError()

    def log(self, name: str, value: Any) -> None:
        # FIXME: this is a workaround to make the log at least show somewhere.
        # Need a refactor in logger to formalize this.
        if isinstance(value, (np.ndarray, list)):
            value = np.mean(value)
        _logger.info(f"[Iter {self.trainer.current_iter + 1}] {name} = {value}")

    def log_dict(self, data: Dict[str, Any]) -> None:
        for name, value in data.items():
            self.log(name, value)

    def state_dict(self) -> Dict:
        """返回当前容器状态的检查点。"""
        return {"policy": self.policy.state_dict()}

    def load_state_dict(self, state_dict: Dict) -> None:
        """从之前保存的状态字典恢复检查点。"""
        self.policy.load_state_dict(state_dict["policy"])


class TrainingVessel(TrainingVesselBase):
    """训练容器的默认实现。

    ``__init__``接受初始状态序列以便创建迭代器。
    ``train``、``validate``、``test``各执行一次收集(训练中还包含更新)。
    默认情况下，训练初始状态会在训练期间无限重复，
    收集器会控制每次迭代的轮次(episode)数量。
    在验证和测试中，验证/测试初始状态将仅使用一次。

    额外超参数(仅用于训练)包括:

    - ``buffer_size``: 回放缓冲区大小
    - ``episode_per_iter``: 每次训练收集的轮次数量。可被快速开发模式覆盖。
    - ``update_kwargs``: 传递给``policy.update``的关键字参数。
      例如``dict(repeat=10, batch_size=64)``。
    """

    def __init__(
        self,
        *,
        simulator_fn: Callable[[InitialStateType], Simulator[InitialStateType, StateType, ActType]],
        state_interpreter: StateInterpreter[StateType, ObsType],
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        policy: BasePolicy,
        reward: Reward,
        train_initial_states: Sequence[InitialStateType] | None = None,
        val_initial_states: Sequence[InitialStateType] | None = None,
        test_initial_states: Sequence[InitialStateType] | None = None,
        buffer_size: int = 20000,
        episode_per_iter: int = 1000,
        update_kwargs: Dict[str, Any] = cast(Dict[str, Any], None),
    ):
        self.simulator_fn = simulator_fn  # type: ignore
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter
        self.policy = policy
        self.reward = reward
        self.train_initial_states = train_initial_states
        self.val_initial_states = val_initial_states
        self.test_initial_states = test_initial_states
        self.buffer_size = buffer_size
        self.episode_per_iter = episode_per_iter
        self.update_kwargs = update_kwargs or {}

    def train_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        if self.train_initial_states is not None:
            _logger.info("Training initial states collection size: %d", len(self.train_initial_states))
            # Implement fast_dev_run here.
            train_initial_states = self._random_subset("train", self.train_initial_states, self.trainer.fast_dev_run)
            return DataQueue(train_initial_states, repeat=-1, shuffle=True)
        return super().train_seed_iterator()

    def val_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        if self.val_initial_states is not None:
            _logger.info("Validation initial states collection size: %d", len(self.val_initial_states))
            val_initial_states = self._random_subset("val", self.val_initial_states, self.trainer.fast_dev_run)
            return DataQueue(val_initial_states, repeat=1)
        return super().val_seed_iterator()

    def test_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        if self.test_initial_states is not None:
            _logger.info("Testing initial states collection size: %d", len(self.test_initial_states))
            test_initial_states = self._random_subset("test", self.test_initial_states, self.trainer.fast_dev_run)
            return DataQueue(test_initial_states, repeat=1)
        return super().test_seed_iterator()

    def train(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        """创建收集器并收集``episode_per_iter``轮次(episodes)。
        在收集的回放缓冲区上更新策略。
        """
        self.policy.train()

        with vector_env.collector_guard():
            collector = Collector(
                self.policy, vector_env, VectorReplayBuffer(self.buffer_size, len(vector_env)), exploration_noise=True
            )

            # Number of episodes collected in each training iteration can be overridden by fast dev run.
            if self.trainer.fast_dev_run is not None:
                episodes = self.trainer.fast_dev_run
            else:
                episodes = self.episode_per_iter

            col_result = collector.collect(n_episode=episodes)
            update_result = self.policy.update(sample_size=0, buffer=collector.buffer, **self.update_kwargs)
            res = {**col_result, **update_result}
            self.log_dict(res)
            return res

    def validate(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            res = test_collector.collect(n_step=INF * len(vector_env))
            self.log_dict(res)
            return res

    def test(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            res = test_collector.collect(n_step=INF * len(vector_env))
            self.log_dict(res)
            return res

    @staticmethod
    def _random_subset(name: str, collection: Sequence[T], size: int | None) -> Sequence[T]:
        if size is None:
            # Size = None -> original collection
            return collection
        order = np.random.permutation(len(collection))
        res = [collection[o] for o in order[:size]]
        _logger.info(
            "Fast running in development mode. Cut %s initial states from %d to %d.",
            name,
            len(collection),
            len(res),
        )
        return res
