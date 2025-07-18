# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""分布式强化学习日志器。

:class:`LogCollector` 在每个环境工作进程中运行，从模拟器状态收集日志信息，并将其（作为字典）添加到每步返回的辅助信息中。

:class:`LogWriter` 在中央工作进程中运行，解码每个工作进程中由:class:`LogCollector`收集的字典，并将其写入控制台、日志文件或tensorboard等。

这两个模块通过``env.step()``返回的"info"中的"log"字段进行通信。
"""

# 注意：此文件包含许多硬编码/临时规则。
# 重构将是未来的任务之一。

from __future__ import annotations

import logging
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Sequence, Set, Tuple, TypeVar

import numpy as np
import pandas as pd

from qlib.log import get_module_logger

if TYPE_CHECKING:
    from .env_wrapper import InfoDict


__all__ = ["LogCollector", "LogWriter", "LogLevel", "LogBuffer", "ConsoleWriter", "CsvWriter"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LogLevel(IntEnum):
    """强化学习训练的日志级别。
    每个日志级别的处理行为取决于:class:`LogWriter`的实现。
    """

    DEBUG = 10
    """仅在调试模式下查看指标。"""
    PERIODIC = 20
    """定期查看指标。"""
    # FIXME: 对此尚未深入思考，暂保留此迭代版本。

    INFO = 30
    """重要日志消息。"""
    CRITICAL = 40
    """LogWriter应始终处理CRITICAL消息"""


class LogCollector:
    """日志首先在每个环境工作进程中收集，然后在向量环境的中央线程中聚合流。

    在:class:`LogCollector`中，每个指标都被添加到一个字典中，需要在每步调用``reset()``清空。
    该字典通过``env.step()``中的``info``发送，并由向量环境中的:class:`LogWriter`解码。

    ``min_loglevel``用于优化目的：避免网络/管道中的过多流量。
    """

    _logged: Dict[str, Tuple[int, Any]]
    _min_loglevel: int

    def __init__(self, min_loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        self._min_loglevel = int(min_loglevel)

    def reset(self) -> None:
        """清除所有已收集的内容。"""
        self._logged = {}

    def _add_metric(self, name: str, metric: Any, loglevel: int | LogLevel) -> None:
        if name in self._logged:
            raise ValueError(f"A metric with {name} is already added. Please change a name or reset the log collector.")
        self._logged[name] = (int(loglevel), metric)

    def add_string(self, name: str, string: str, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """添加带名称的字符串到日志内容中。"""
        if loglevel < self._min_loglevel:
            return
        if not isinstance(string, str):
            raise TypeError(f"{string} 不是字符串。")
        self._add_metric(name, string, loglevel)

    def add_scalar(self, name: str, scalar: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """添加带名称的标量到日志内容中。
        标量将被转换为浮点数。
        """
        if loglevel < self._min_loglevel:
            return

        if hasattr(scalar, "item"):
            # 可能是单元素数字
            scalar = scalar.item()
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"{scalar} 不是且无法转换为浮点数或整数。")
        scalar = float(scalar)
        self._add_metric(name, scalar, loglevel)

    def add_array(
        self,
        name: str,
        array: np.ndarray | pd.DataFrame | pd.Series,
        loglevel: int | LogLevel = LogLevel.PERIODIC,
    ) -> None:
        """添加带名称的数组到日志中。"""
        if loglevel < self._min_loglevel:
            return

        if not isinstance(array, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(f"{array} 不是ndarray、DataFrame或Series类型。")
        self._add_metric(name, array, loglevel)

    def add_any(self, name: str, obj: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """记录任意类型的对象。

        由于是"任意"对象，唯一能接受它的LogWriter是pickle。
        因此，pickle必须能够序列化它。
        """
        if loglevel < self._min_loglevel:
            return

        # FIXME: 检测并处理可能是标量或数组的对象

        self._add_metric(name, obj, loglevel)

    def logs(self) -> Dict[str, np.ndarray]:
        return {key: np.asanyarray(value, dtype="object") for key, value in self._logged.items()}


class LogWriter(Generic[ObsType, ActType]):
    """日志写入器基类，由有限环境在每次重置和步骤时触发。

    如何处理特定日志取决于子类化:class:`LogWriter`的实现。
    一般原则是，它应该处理高于其日志级别（包括）的日志，并丢弃不可接受的日志。例如，控制台日志器显然无法处理图像。
    """

    episode_count: int
    """回合计数器。"""

    step_count: int
    """步数计数器。"""

    global_step: int
    """全局步数计数器。在``clear``中不会被清除。"""

    global_episode: int
    """全局回合计数器。在``clear``中不会被清除。"""

    active_env_ids: Set[int]
    """向量环境中活跃的环境ID集合。"""

    episode_lengths: Dict[int, int]
    """从环境ID到回合长度的映射。"""

    episode_rewards: Dict[int, List[float]]
    """从环境ID到回合总奖励的映射。"""

    episode_logs: Dict[int, list]
    """从环境ID到回合日志的映射。"""

    def __init__(self, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        self.loglevel = loglevel

        self.global_step = 0
        self.global_episode = 0

        # Information, logs of one episode is stored here.
        # This assumes that episode is not too long to fit into the memory.
        self.episode_lengths = dict()
        self.episode_rewards = dict()
        self.episode_logs = dict()

        self.clear()

    def clear(self):
        """Clear all the metrics for a fresh start.
        To make the logger instance reusable.
        """
        self.episode_count = self.step_count = 0
        self.active_env_ids = set()

    def state_dict(self) -> dict:
        """Save the states of the logger to a dict."""
        return {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "global_step": self.global_step,
            "global_episode": self.global_episode,
            "active_env_ids": self.active_env_ids,
            "episode_lengths": self.episode_lengths,
            "episode_rewards": self.episode_rewards,
            "episode_logs": self.episode_logs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the states of current logger from a dict."""
        self.episode_count = state_dict["episode_count"]
        self.step_count = state_dict["step_count"]
        self.global_step = state_dict["global_step"]
        self.global_episode = state_dict["global_episode"]

        # These are runtime infos.
        # Though they are loaded, I don't think it really helps.
        self.active_env_ids = state_dict["active_env_ids"]
        self.episode_lengths = state_dict["episode_lengths"]
        self.episode_rewards = state_dict["episode_rewards"]
        self.episode_logs = state_dict["episode_logs"]

    @staticmethod
    def aggregation(array: Sequence[Any], name: str | None = None) -> Any:
        """Aggregation function from step-wise to episode-wise.

        If it's a sequence of float, take the mean.
        Otherwise, take the first element.

        If a name is specified and,

        - if it's ``reward``, the reduction will be sum.
        """
        assert len(array) > 0, "The aggregated array must be not empty."
        if all(isinstance(v, float) for v in array):
            if name == "reward":
                return np.sum(array)
            return np.mean(array)
        else:
            return array[0]

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        """在每个轨迹结束时触发。

        参数
        ----------
        length
            此轨迹的长度。
        rewards
            本回合每步奖励的列表。
        contents
            每步的日志内容。
        """

    def log_step(self, reward: float, contents: Dict[str, Any]) -> None:
        """在每步触发。

        参数
        ----------
        reward
            此步的奖励。
        contents
            此步的日志内容。
        """

    def on_env_step(self, env_id: int, obs: ObsType, rew: float, done: bool, info: InfoDict) -> None:
        """Callback for finite env, on each step."""

        # Update counter
        self.global_step += 1
        self.step_count += 1

        self.active_env_ids.add(env_id)
        self.episode_lengths[env_id] += 1
        # TODO: reward can be a list of list for MARL
        self.episode_rewards[env_id].append(rew)

        values: Dict[str, Any] = {}

        for key, (loglevel, value) in info["log"].items():
            if loglevel >= self.loglevel:  # FIXME: this is actually incorrect (see last FIXME)
                values[key] = value
        self.episode_logs[env_id].append(values)

        self.log_step(rew, values)

        if done:
            # Update counter
            self.global_episode += 1
            self.episode_count += 1

            self.log_episode(self.episode_lengths[env_id], self.episode_rewards[env_id], self.episode_logs[env_id])

    def on_env_reset(self, env_id: int, _: ObsType) -> None:
        """有限环境的回调函数。

        重置回合统计信息。由于tianshou的限制<https://github.com/thu-ml/tianshou/issues/605>，此处不记录任何任务特定信息。
        """
        self.episode_lengths[env_id] = 0
        self.episode_rewards[env_id] = []
        self.episode_logs[env_id] = []

    def on_env_all_ready(self) -> None:
        """当所有环境准备就绪可以运行时调用。
        通常，日志器应在此处重置。
        """
        self.clear()

    def on_env_all_done(self) -> None:
        """所有操作完成，进行清理工作。"""


class LogBuffer(LogWriter):
    """将所有数字保存在内存中。

    无法聚合的对象（如字符串、张量、图像）不能存储在缓冲区中。
    要持久化它们，请使用:class:`PickleWriter`。

    每次日志缓冲区收到新指标时，都会触发回调，
    这在训练器内部跟踪指标时非常有用。

    参数
    ----------
    callback
        接收三个参数的回调函数：

        - on_episode: 是否在回合结束时调用
        - on_collect: 是否在收集结束时调用
        - log_buffer: :class:`LogBbuffer`对象

        不需要返回值。
    """

    # FIXME: needs a metric count

    def __init__(self, callback: Callable[[bool, bool, LogBuffer], None], loglevel: int | LogLevel = LogLevel.PERIODIC):
        super().__init__(loglevel)
        self.callback = callback

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "latest_metrics": self._latest_metrics,
            "aggregated_metrics": self._aggregated_metrics,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._latest_metrics = state_dict["latest_metrics"]
        self._aggregated_metrics = state_dict["aggregated_metrics"]
        return super().load_state_dict(state_dict)

    def clear(self):
        super().clear()
        self._latest_metrics: dict[str, float] | None = None
        self._aggregated_metrics: dict[str, float] = defaultdict(float)

    def log_episode(self, length: int, rewards: list[float], contents: list[dict[str, Any]]) -> None:
        # FIXME Dup of ConsoleWriter
        episode_wise_contents: dict[str, list] = defaultdict(list)
        for step_contents in contents:
            for name, value in step_contents.items():
                # FIXME This could be false-negative for some numpy types
                if isinstance(value, float):
                    episode_wise_contents[name].append(value)

        logs: dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore
            self._aggregated_metrics[name] += logs[name]

        self._latest_metrics = logs

        self.callback(True, False, self)

    def on_env_all_done(self) -> None:
        # This happens when collect exits
        self.callback(False, True, self)

    def episode_metrics(self) -> dict[str, float]:
        """Retrieve the numeric metrics of the latest episode."""
        if self._latest_metrics is None:
            raise ValueError("No episode metrics available yet.")
        return self._latest_metrics

    def collect_metrics(self) -> dict[str, float]:
        """Retrieve the aggregated metrics of the latest collect."""
        return {name: value / self.episode_count for name, value in self._aggregated_metrics.items()}


class ConsoleWriter(LogWriter):
    """定期将日志消息写入控制台。

    它为每个指标跟踪一个平均计量器，即从上次``clear()``到现在的平均值。
    每个指标的显示格式为``<名称> <最新值> (<平均值>)``。

    非单一数字指标会自动跳过。


    prefix: str
        Prefix can be set via ``writer.prefix``.

    """
    def __init__(
        self,
        log_every_n_episode: int = 20,
        total_episodes: int | None = None,
        float_format: str = ":.4f",
        counter_format: str = ":4d",
        loglevel: int | LogLevel = LogLevel.PERIODIC,
    ) -> None:
        super().__init__(loglevel)
        # TODO: 支持按步数记录日志
        self.log_every_n_episode = log_every_n_episode
        self.total_episodes = total_episodes

        self.counter_format = counter_format
        self.float_format = float_format

        self.prefix = ""

        self.console_logger = get_module_logger(__name__, level=logging.INFO)

    # FIXME: save & reload

    def clear(self) -> None:
        super().clear()
        # Clear average meters
        self.metric_counts: Dict[str, int] = defaultdict(int)
        self.metric_sums: Dict[str, float] = defaultdict(float)

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        # Aggregate step-wise to episode-wise
        episode_wise_contents: Dict[str, list] = defaultdict(list)

        for step_contents in contents:
            for name, value in step_contents.items():
                if isinstance(value, float):
                    episode_wise_contents[name].append(value)

        # Generate log contents and track them in average-meter.
        # This should be done at every step, regardless of periodic or not.
        logs: Dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore

        for name, value in logs.items():
            self.metric_counts[name] += 1
            self.metric_sums[name] += value

        if self.episode_count % self.log_every_n_episode == 0 or self.episode_count == self.total_episodes:
            # Only log periodically or at the end
            self.console_logger.info(self.generate_log_message(logs))

    def generate_log_message(self, logs: Dict[str, float]) -> str:
        if self.prefix:
            msg_prefix = self.prefix + " "
        else:
            msg_prefix = ""
        if self.total_episodes is None:
            msg_prefix += "[Step {" + self.counter_format + "}]"
        else:
            msg_prefix += "[{" + self.counter_format + "}/" + str(self.total_episodes) + "]"
        msg_prefix = msg_prefix.format(self.episode_count)

        msg = ""
        for name, value in logs.items():
            # Double-space as delimiter
            format_template = r"  {} {" + self.float_format + "} ({" + self.float_format + "})"
            msg += format_template.format(name, value, self.metric_sums[name] / self.metric_counts[name])

        msg = msg_prefix + " " + msg

        return msg


class CsvWriter(LogWriter):
    """将所有回合指标转储到``result.csv``文件中。

    这不是正确的实现，仅用于第一次迭代。
    """

    SUPPORTED_TYPES = (float, str, pd.Timestamp)

    all_records: List[Dict[str, Any]]

    # FIXME: save & reload

    def __init__(self, output_dir: Path, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        super().__init__(loglevel)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def clear(self) -> None:
        super().clear()
        self.all_records = []

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        # FIXME 与ConsoleLogger相同，需要重构以消除代码重复
        episode_wise_contents: Dict[str, list] = defaultdict(list)

        for step_contents in contents:
            for name, value in step_contents.items():
                if isinstance(value, self.SUPPORTED_TYPES):
                    episode_wise_contents[name].append(value)

        logs: Dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore

        self.all_records.append(logs)

    def on_env_all_done(self) -> None:
        # FIXME: this is temporary
        pd.DataFrame.from_records(self.all_records).to_csv(self.output_dir / "result.csv", index=False)


# The following are not implemented yet.


class PickleWriter(LogWriter):
    """将日志转储到pickle文件。"""


class TensorboardWriter(LogWriter):
    """将日志写入可通过tensorboard可视化的事件文件。"""


class MlflowWriter(LogWriter):
    """将日志添加到mlflow。"""
