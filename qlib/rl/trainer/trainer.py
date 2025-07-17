# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
import copy
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, OrderedDict, Sequence, TypeVar, cast

import torch

from qlib.log import get_module_logger
from qlib.rl.simulator import InitialStateType
from qlib.rl.utils import EnvWrapper, FiniteEnvType, LogBuffer, LogCollector, LogLevel, LogWriter, vectorize_env
from qlib.rl.utils.finite_env import FiniteVectorEnv
from qlib.typehint import Literal

from .callbacks import Callback
from .vessel import TrainingVesselBase

_logger = get_module_logger(__name__)


T = TypeVar("T")


class Trainer:
    """
    用于在特定任务上训练策略的工具。

    与传统深度学习训练器不同，此训练器的迭代单位是"收集"(collect)，
    而非"epoch"或"mini-batch"。
    每次收集时，:class:`Collector`会收集一定数量的策略-环境交互数据，
    并累积到回放缓冲区中。此缓冲区用作训练策略的"数据"。
    每次收集结束时，策略会被*更新*多次。

    API与`PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`__有些相似，
    但由于此训练器专为RL应用构建，大多数配置都在RL上下文中，
    因此本质上不同。
    我们仍在寻找整合现有训练器库的方法，因为构建与这些库同样强大的训练器
    需要大量工作，且这不是我们的主要目标。

    与`tianshou的内置训练器 <https://tianshou.readthedocs.io/en/master/api/tianshou.trainer.html>`__
    也完全不同，因为此实现要复杂得多。

    参数
    ----------
    max_iters
        停止前的最大迭代次数。
    val_every_n_iters
        每n次迭代(即训练收集)执行一次验证。
    logger
        记录回测结果的日志记录器。必须提供日志记录器，
        否则所有信息都将丢失。
    finite_env_type
        有限环境实现类型。
    concurrency
        并行工作进程数。
    fast_dev_run
        创建用于调试的子集。
        具体实现取决于训练容器的实现方式。
        对于:class:`~qlib.rl.vessel.TrainingVessel`，如果大于零，
        将使用大小为``fast_dev_run``的随机子集
        替代``train_initial_states``和``val_initial_states``。
    """

    should_stop: bool
    """设置为true可停止训练。"""

    metrics: dict
    """训练/验证/测试中产生的数值指标。
    在训练/验证过程中，指标来自最新的一轮(episode)。
    当每次训练/验证迭代完成时，指标将是该迭代中
    所有轮次(episode)的聚合结果。

    每次新训练迭代开始时会被清空。

    在fit过程中，验证指标会以``val/``为前缀。
    """

    current_iter: int
    """当前训练迭代(收集)次数。"""

    loggers: List[LogWriter]
    """日志记录器列表。"""

    def __init__(
        self,
        *,
        max_iters: int | None = None,
        val_every_n_iters: int | None = None,
        loggers: LogWriter | List[LogWriter] | None = None,
        callbacks: List[Callback] | None = None,
        finite_env_type: FiniteEnvType = "subproc",
        concurrency: int = 2,
        fast_dev_run: int | None = None,
    ):
        self.max_iters = max_iters
        self.val_every_n_iters = val_every_n_iters

        if isinstance(loggers, list):
            self.loggers = loggers
        elif isinstance(loggers, LogWriter):
            self.loggers = [loggers]
        else:
            self.loggers = []

        self.loggers.append(LogBuffer(self._metrics_callback, loglevel=self._min_loglevel()))

        self.callbacks: List[Callback] = callbacks if callbacks is not None else []
        self.finite_env_type = finite_env_type
        self.concurrency = concurrency
        self.fast_dev_run = fast_dev_run

        self.current_stage: Literal["train", "val", "test"] = "train"

        self.vessel: TrainingVesselBase = cast(TrainingVesselBase, None)

    def initialize(self):
        """初始化整个训练过程。

        此处的状态应与state_dict保持同步。
        """
        self.should_stop = False
        self.current_iter = 0
        self.current_episode = 0
        self.current_stage = "train"

    def initialize_iter(self):
        """初始化一次迭代/收集。"""
        self.metrics = {}

    def state_dict(self) -> dict:
        """尽可能将当前训练的所有状态存入字典。

        它不会尝试处理一次训练收集中间可能出现的所有状态类型。
        对于大多数情况，在每次迭代结束时，结果通常是正确的。

        注意，收集器中回放缓冲区数据的丢失是预期行为。
        """
        return {
            "vessel": self.vessel.state_dict(),
            "callbacks": {name: callback.state_dict() for name, callback in self.named_callbacks().items()},
            "loggers": {name: logger.state_dict() for name, logger in self.named_loggers().items()},
            "should_stop": self.should_stop,
            "current_iter": self.current_iter,
            "current_episode": self.current_episode,
            "current_stage": self.current_stage,
            "metrics": self.metrics,
        }

    @staticmethod
    def get_policy_state_dict(ckpt_path: Path) -> OrderedDict:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "vessel" in state_dict:
            state_dict = state_dict["vessel"]["policy"]
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """将所有状态加载到当前训练器中。"""
        self.vessel.load_state_dict(state_dict["vessel"])
        for name, callback in self.named_callbacks().items():
            callback.load_state_dict(state_dict["callbacks"][name])
        for name, logger in self.named_loggers().items():
            logger.load_state_dict(state_dict["loggers"][name])
        self.should_stop = state_dict["should_stop"]
        self.current_iter = state_dict["current_iter"]
        self.current_episode = state_dict["current_episode"]
        self.current_stage = state_dict["current_stage"]
        self.metrics = state_dict["metrics"]

    def named_callbacks(self) -> Dict[str, Callback]:
        """获取带有名称的回调函数集合。
        在保存检查点时很有用。
        """
        return _named_collection(self.callbacks)

    def named_loggers(self) -> Dict[str, LogWriter]:
        """获取带有名称的日志记录器集合。
        在保存检查点时很有用。
        """
        return _named_collection(self.loggers)

    def fit(self, vessel: TrainingVesselBase, ckpt_path: Path | None = None) -> None:
        """在定义的模拟器上训练RL策略。

        参数
        ----------
        vessel
            训练中使用的所有元素的集合。
        ckpt_path
            加载预pre-trained / paused的训练检查点。
        """
        self.vessel = vessel
        vessel.assign_trainer(self)

        if ckpt_path is not None:
            _logger.info("Resuming states from %s", str(ckpt_path))
            self.load_state_dict(torch.load(ckpt_path, weights_only=False))
        else:
            self.initialize()

        self._call_callback_hooks("on_fit_start")

        while not self.should_stop:
            msg = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tTrain iteration {self.current_iter + 1}/{self.max_iters}"
            _logger.info(msg)

            self.initialize_iter()

            self._call_callback_hooks("on_iter_start")

            self.current_stage = "train"
            self._call_callback_hooks("on_train_start")

            # TODO
            # Add a feature that supports reloading the training environment every few iterations.
            with _wrap_context(vessel.train_seed_iterator()) as iterator:
                vector_env = self.venv_from_iterator(iterator)
                self.vessel.train(vector_env)
                del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.

            self._call_callback_hooks("on_train_end")

            if self.val_every_n_iters is not None and (self.current_iter + 1) % self.val_every_n_iters == 0:
                # Implementation of validation loop
                self.current_stage = "val"
                self._call_callback_hooks("on_validate_start")
                with _wrap_context(vessel.val_seed_iterator()) as iterator:
                    vector_env = self.venv_from_iterator(iterator)
                    self.vessel.validate(vector_env)
                    del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.

                self._call_callback_hooks("on_validate_end")

            # This iteration is considered complete.
            # Bumping the current iteration counter.
            self.current_iter += 1

            if self.max_iters is not None and self.current_iter >= self.max_iters:
                self.should_stop = True

            self._call_callback_hooks("on_iter_end")

        self._call_callback_hooks("on_fit_end")

    def test(self, vessel: TrainingVesselBase) -> None:
        """在模拟器上测试RL策略。

        模拟器将使用``test_seed_iterator``生成的数据。

        参数
        ----------
        vessel
            所有相关元素的集合。
        """
        self.vessel = vessel
        vessel.assign_trainer(self)

        self.initialize_iter()

        self.current_stage = "test"
        self._call_callback_hooks("on_test_start")
        with _wrap_context(vessel.test_seed_iterator()) as iterator:
            vector_env = self.venv_from_iterator(iterator)
            self.vessel.test(vector_env)
            del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.
        self._call_callback_hooks("on_test_end")

    def venv_from_iterator(self, iterator: Iterable[InitialStateType]) -> FiniteVectorEnv:
        """从迭代器和训练容器创建向量化环境。"""

        def env_factory():
            # FIXME: state_interpreter and action_interpreter are stateful (having a weakref of env),
            # and could be thread unsafe.
            # I'm not sure whether it's a design flaw.
            # I'll rethink about this when designing the trainer.

            if self.finite_env_type == "dummy":
                # We could only experience the "threading-unsafe" problem in dummy.
                state = copy.deepcopy(self.vessel.state_interpreter)
                action = copy.deepcopy(self.vessel.action_interpreter)
                rew = copy.deepcopy(self.vessel.reward)
            else:
                state = self.vessel.state_interpreter
                action = self.vessel.action_interpreter
                rew = self.vessel.reward

            return EnvWrapper(
                self.vessel.simulator_fn,
                state,
                action,
                iterator,
                rew,
                logger=LogCollector(min_loglevel=self._min_loglevel()),
            )

        return vectorize_env(
            env_factory,
            self.finite_env_type,
            self.concurrency,
            self.loggers,
        )

    def _metrics_callback(self, on_episode: bool, on_collect: bool, log_buffer: LogBuffer) -> None:
        if on_episode:
            # Update the global counter.
            self.current_episode = log_buffer.global_episode
            metrics = log_buffer.episode_metrics()
        elif on_collect:
            # Update the latest metrics.
            metrics = log_buffer.collect_metrics()
        if self.current_stage == "val":
            metrics = {"val/" + name: value for name, value in metrics.items()}
        self.metrics.update(metrics)

    def _call_callback_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            fn(self, self.vessel, *args, **kwargs)

    def _min_loglevel(self):
        if not self.loggers:
            return LogLevel.PERIODIC
        else:
            # To save bandwidth
            return min(lg.loglevel for lg in self.loggers)


@contextmanager
def _wrap_context(obj):
    """Make any object a (possibly dummy) context manager."""

    if isinstance(obj, AbstractContextManager):
        # obj has __enter__ and __exit__
        with obj as ctx:
            yield ctx
    else:
        yield obj


def _named_collection(seq: Sequence[T]) -> Dict[str, T]:
    """Convert a list into a dict, where each item is named with its type."""
    res = {}
    retry_cnt: collections.Counter = collections.Counter()
    for item in seq:
        typename = type(item).__name__.lower()
        key = typename if retry_cnt[typename] == 0 else f"{typename}{retry_cnt[typename]}"
        retry_cnt[typename] += 1
        res[key] = item
    return res
