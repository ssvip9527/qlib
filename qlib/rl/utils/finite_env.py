# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
支持向量环境中的有限环境。
详见 https://github.com/thu-ml/tianshou/issues/322
"""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union, cast

import gym
import numpy as np
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from qlib.typehint import Literal

from .log import LogWriter

__all__ = [
    "generate_nan_observation",
    "check_nan_observation",
    "FiniteVectorEnv",
    "FiniteDummyVectorEnv",
    "FiniteSubprocVectorEnv",
    "FiniteShmemVectorEnv",
    "FiniteEnvType",
    "vectorize_env",
]

FiniteEnvType = Literal["dummy", "subproc", "shmem"]
T = Union[dict, list, tuple, np.ndarray]


def fill_invalid(obj: int | float | bool | T) -> T:
    if isinstance(obj, (int, float, bool)):
        return fill_invalid(np.array(obj))
    if hasattr(obj, "dtype"):
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating):
                return np.full_like(obj, np.nan)
            return np.full_like(obj, np.iinfo(obj.dtype).max)
        # 处理tianshou的sharray不支持numpy数字的边缘情况
        return fill_invalid(np.array(obj))
    elif isinstance(obj, dict):
        return {k: fill_invalid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fill_invalid(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(fill_invalid(v) for v in obj)
    raise ValueError(f"Unsupported value to fill with invalid: {obj}")


def is_invalid(arr: int | float | bool | T) -> bool:
    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.floating):
            return np.isnan(arr).all()
        return cast(bool, cast(np.ndarray, np.iinfo(arr.dtype).max == arr).all())
    if isinstance(arr, dict):
        return all(is_invalid(o) for o in arr.values())
    if isinstance(arr, (list, tuple)):
        return all(is_invalid(o) for o in arr)
    if isinstance(arr, (int, float, bool, np.number)):
        return is_invalid(np.array(arr))
    return True


def generate_nan_observation(obs_space: gym.Space) -> Any:
    """表示环境未接收种子的NaN观测值。

    我们假设观测值复杂且必须包含浮点类型，
    否则此逻辑将无法工作。
    """

    sample = obs_space.sample()
    sample = fill_invalid(sample)
    return sample


def check_nan_observation(obs: Any) -> bool:
    """检查观测值是否由:func:`generate_nan_observation`生成。"""
    return is_invalid(obs)


class FiniteVectorEnv(BaseVectorEnv):
    """允许并行环境工作进程共享单个DataQueue直到耗尽。

    参见`tianshou issue #322 <https://github.com/thu-ml/tianshou/issues/322>`_。

    需求是确保每个可能的种子(在我们的案例中存储在:class:`qlib.rl.utils.DataQueue`中)
    被恰好一个环境消费。tianshou原生VectorEnv和Collector无法实现这一点，
    因为tianshou不知道"恰好一个"的约束，可能会启动额外工作进程。

    考虑一个边界情况：并发数为2但DataQueue中只有一个种子。
    根据collect逻辑必须调用两个工作进程的reset。
    无论结果如何都会收集两个工作进程的返回结果。
    问题在于其中一个reset结果必然是无效或重复的，
    因为队列中只有一个需求，而collector不知道这种情况。

    幸运的是，我们可以修改向量环境，在单个环境和向量环境间建立协议。
    单个环境(在我们的案例中应为:class:`qlib.rl.utils.EnvWrapper`)负责
    从队列读取，并在队列耗尽时生成特殊观测值。这种特殊观测值
    称为"nan观测值"，因为在共享内存向量环境中直接使用none会导致问题。
    :class:`FiniteVectorEnv`然后从所有工作进程读取观测值，并选择非nan的
    观测值。它还维护``_alive_env_ids``来追踪哪些工作进程不应再被调用。
    当所有环境都耗尽时，它将抛出StopIteration异常。

    在collector中使用此向量环境有两种情况：

    1. 如果数据队列有限(通常在推理时)，collector应收集"无限"数量的
       轮次(episodes)，直到向量环境自行耗尽。
    2. 如果数据队列无限(通常在训练时)，collector可以设置轮次/步数。
       这种情况下数据将随机排序，一些重复无关紧要。

    此向量环境的一个额外功能是具有显式收集子工作进程日志的日志记录器。
    参见:class:`qlib.rl.utils.LogWriter`。
    """

    _logger: list[LogWriter]

    def __init__(
        self, logger: LogWriter | list[LogWriter] | None, env_fns: list[Callable[..., gym.Env]], **kwargs: Any
    ) -> None:
        super().__init__(env_fns, **kwargs)

        if isinstance(logger, list):
            self._logger = logger
        elif isinstance(logger, LogWriter):
            self._logger = [logger]
        else:
            self._logger = []
        self._alive_env_ids: Set[int] = set()
        self._reset_alive_envs()
        self._default_obs = self._default_info = self._default_rew = None
        self._zombie = False

        self._collector_guarded: bool = False

    def _reset_alive_envs(self) -> None:
        if not self._alive_env_ids:
            # 启动或耗尽时
            self._alive_env_ids = set(range(self.env_num))

    # 解决tianshou的缓冲区和批次问题
    def _set_default_obs(self, obs: Any) -> None:
        if obs is not None and self._default_obs is None:
            self._default_obs = copy.deepcopy(obs)

    def _set_default_info(self, info: Any) -> None:
        if info is not None and self._default_info is None:
            self._default_info = copy.deepcopy(info)

    def _set_default_rew(self, rew: Any) -> None:
        if rew is not None and self._default_rew is None:
            self._default_rew = copy.deepcopy(rew)

    def _get_default_obs(self) -> Any:
        return copy.deepcopy(self._default_obs)

    def _get_default_info(self) -> Any:
        return copy.deepcopy(self._default_info)

    def _get_default_rew(self) -> Any:
        return copy.deepcopy(self._default_rew)

    # 结束

    @staticmethod
    def _postproc_env_obs(obs: Any) -> Optional[Any]:
        # 为共享内存向量环境保留，用于恢复空观测值
        if obs is None or check_nan_observation(obs):
            return None
        return obs

    @contextmanager
    def collector_guard(self) -> Generator[FiniteVectorEnv, None, None]:
        """保护收集器。建议每次收集时都使用保护。

        此保护有两个目的：

        1. 捕获并忽略StopIteration异常，这是FiniteEnv抛出的停止信号，
           用于通知tianshou ``collector.collect()`` 应该退出。
        2. 通知日志记录器收集已准备好/已完成。

        示例
        ----
        >>> with finite_env.collector_guard():
        ...     collector.collect(n_episode=INF)
        """
        self._collector_guarded = True

        for logger in self._logger:
            logger.on_env_all_ready()

        try:
            yield self
        except StopIteration:
            pass
        finally:
            self._collector_guarded = False

        # At last trigger the loggers
        for logger in self._logger:
            logger.on_env_all_done()

    def reset(
        self,
        id: int | List[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        assert not self._zombie

        # 检查是否由collector_guard()保护
        if not self._collector_guarded:
            warnings.warn(
                "Collector is not guarded by FiniteEnv. "
                "This may cause unexpected problems, like unexpected StopIteration exception, "
                "or missing logs.",
                RuntimeWarning,
            )

        wrapped_id = self._wrap_id(id)
        self._reset_alive_envs()

        # 请求父类重置活跃环境并重新映射到当前索引
        request_id = [i for i in wrapped_id if i in self._alive_env_ids]
        obs = [None] * len(wrapped_id)
        id2idx = {i: k for k, i in enumerate(wrapped_id)}
        if request_id:
            for i, o in zip(request_id, super().reset(request_id)):
                obs[id2idx[i]] = self._postproc_env_obs(o)

        for i, o in zip(wrapped_id, obs):
            if o is None and i in self._alive_env_ids:
                self._alive_env_ids.remove(i)

        # 日志记录
        for i, o in zip(wrapped_id, obs):
            if i in self._alive_env_ids:
                for logger in self._logger:
                    logger.on_env_reset(i, obs)

        # fill empty observation with default(fake) observation
        for o in obs:
            self._set_default_obs(o)
        for i, o in enumerate(obs):
            if o is None:
                obs[i] = self._get_default_obs()

        if not self._alive_env_ids:
            # comment this line so that the env becomes indispensable
            # self.reset()
            self._zombie = True
            raise StopIteration

        return np.stack(obs)

    def step(
        self,
        action: np.ndarray,
        id: int | List[int] | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert not self._zombie
        wrapped_id = self._wrap_id(id)
        id2idx = {i: k for k, i in enumerate(wrapped_id)}
        request_id = list(filter(lambda i: i in self._alive_env_ids, wrapped_id))
        result = [[None, None, False, None] for _ in range(len(wrapped_id))]

        # 请求父类处理活跃环境的step并重新映射到当前索引
        if request_id:
            valid_act = np.stack([action[id2idx[i]] for i in request_id])
            for i, r in zip(request_id, zip(*super().step(valid_act, request_id))):
                result[id2idx[i]] = list(r)
                result[id2idx[i]][0] = self._postproc_env_obs(result[id2idx[i]][0])

        # logging
        for i, r in zip(wrapped_id, result):
            if i in self._alive_env_ids:
                for logger in self._logger:
                    logger.on_env_step(i, *r)

        # fill empty observation/info with default(fake)
        for _, r, ___, i in result:
            self._set_default_info(i)
            self._set_default_rew(r)
        for i, r in enumerate(result):
            if r[0] is None:
                result[i][0] = self._get_default_obs()
            if r[1] is None:
                result[i][1] = self._get_default_rew()
            if r[3] is None:
                result[i][3] = self._get_default_info()

        ret = list(map(np.stack, zip(*result)))
        return cast(Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ret)


class FiniteDummyVectorEnv(FiniteVectorEnv, DummyVectorEnv):
    pass


class FiniteSubprocVectorEnv(FiniteVectorEnv, SubprocVectorEnv):
    pass


class FiniteShmemVectorEnv(FiniteVectorEnv, ShmemVectorEnv):
    pass


def vectorize_env(
    env_factory: Callable[..., gym.Env],
    env_type: FiniteEnvType,
    concurrency: int,
    logger: LogWriter | List[LogWriter],
) -> FiniteVectorEnv:
    """创建向量环境的辅助函数。可用于替换常规的VectorEnv。

    例如，如果你曾经这样写：::

        DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])

    现在你可以替换为：::

        finite_env_factory(lambda: gym.make(task), "dummy", env_num, my_logger)

    通过这样的替换，你将获得两个额外的功能（与普通VectorEnv相比）：

    1. 向量环境将检查NaN观测值，并在发现时终止工作进程。
       有关原因，请参见:class:`FiniteVectorEnv`。
    2. 一个显式收集环境工作进程日志的日志记录器。

    参数
    ----------
    env_factory
        用于实例化单个``gym.Env``的可调用对象。
        所有并发工作进程将使用相同的``env_factory``。
    env_type
        dummy或subproc或shmem。对应于
        `tianshou中的并行方式 <https://tianshou.readthedocs.io/en/master/api/tianshou.env.html#vectorenv>`_。
    concurrency
        并发环境工作进程数。
    logger
        日志记录器。

    警告
    --------
    请勿在此处为``env_factory``使用lambda表达式，因为这可能会创建不正确共享的实例。

    不要这样做：::

        vectorize_env(lambda: EnvWrapper(...), ...)

    请这样做：::

        def env_factory(): ...
        vectorize_env(env_factory, ...)
    """
    env_type_cls_mapping: Dict[str, Type[FiniteVectorEnv]] = {
        "dummy": FiniteDummyVectorEnv,
        "subproc": FiniteSubprocVectorEnv,
        "shmem": FiniteShmemVectorEnv,
    }

    finite_env_cls = env_type_cls_mapping[env_type]

    return finite_env_cls(logger, [env_factory for _ in range(concurrency)])
