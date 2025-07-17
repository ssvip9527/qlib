# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""在训练过程中插入自定义逻辑的回调函数。
模仿Keras/PyTorch-Lightning的钩子机制，但针对RL场景定制。
"""

from __future__ import annotations

import copy
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from qlib.log import get_module_logger
from qlib.typehint import Literal

if TYPE_CHECKING:
    from .trainer import Trainer
    from .vessel import TrainingVesselBase

_logger = get_module_logger(__name__)


class Callback:
    """所有回调函数的基类。"""

    def on_fit_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """在整个训练过程开始前调用。"""

    def on_fit_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """在整个训练过程结束后调用。"""

    def on_train_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """每次训练数据收集开始时调用。"""

    def on_train_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """训练结束时调用。
        要访问训练期间产生的所有输出，可在trainer或vessel中缓存数据，
        并在此钩子中进行后处理。
        """

    def on_validate_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """每次验证运行开始时调用。"""

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """验证结束时调用。"""

    def on_test_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """每次测试运行时调用。"""

    def on_test_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """测试结束时调用。"""

    def on_iter_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """每次迭代(即数据收集)开始时调用。"""

    def on_iter_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """每次迭代结束时调用。
        在``current_iter``递增**之后**调用，
        表示前一次迭代已完成。
        """

    def state_dict(self) -> Any:
        """获取回调函数的状态字典用于暂停和恢复。"""

    def load_state_dict(self, state_dict: Any) -> None:
        """从保存的状态字典恢复回调函数。"""


class EarlyStopping(Callback):
    """当监控指标停止改善时停止训练。

    每次验证结束时触发早停回调。
    它会检查验证产生的指标，
    获取名为``monitor``的指标(默认为``reward``)，
    判断其是否不再增加/减少。
    根据``min_delta``和``patience``参数决定是否停止。
    如果发现指标不再改善，
    则设置``trainer.should_stop``为true，
    终止训练过程。

    实现参考: https://github.com/keras-team/keras/blob/v2.9.0/keras/callbacks.py#L1744-L1893
    """

    def __init__(
        self,
        monitor: str = "reward",
        min_delta: float = 0.0,
        patience: int = 0,
        mode: Literal["min", "max"] = "max",
        baseline: float | None = None,
        restore_best_weights: bool = False,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.restore_best_weights = restore_best_weights
        self.best_weights: Any | None = None

        if mode not in ["min", "max"]:
            raise ValueError("Unsupported earlystopping mode: " + mode)

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def state_dict(self) -> dict:
        return {"wait": self.wait, "best": self.best, "best_weights": self.best_weights, "best_iter": self.best_iter}

    def load_state_dict(self, state_dict: dict) -> None:
        self.wait = state_dict["wait"]
        self.best = state_dict["best"]
        self.best_weights = state_dict["best_weights"]
        self.best_iter = state_dict["best_iter"]

    def on_fit_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.best_weights = None
        self.best_iter = 0

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        current = self.get_monitor_value(trainer)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first iteration if no progress is ever made.
            self.best_weights = copy.deepcopy(vessel.state_dict())

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_iter = trainer.current_iter
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(vessel.state_dict())
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        msg = (
            f"#{trainer.current_iter} current reward: {current:.4f}, best reward: {self.best:.4f} in #{self.best_iter}"
        )
        _logger.info(msg)

        # Only check after the first epoch.
        if self.wait >= self.patience and trainer.current_iter > 0:
            trainer.should_stop = True
            _logger.info(f"On iteration %d: early stopping", trainer.current_iter + 1)
            if self.restore_best_weights and self.best_weights is not None:
                _logger.info("Restoring model weights from the end of the best iteration: %d", self.best_iter + 1)
                vessel.load_state_dict(self.best_weights)

    def get_monitor_value(self, trainer: Trainer) -> Any:
        monitor_value = trainer.metrics.get(self.monitor)
        if monitor_value is None:
            _logger.warning(
                "Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(trainer.metrics.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class MetricsWriter(Callback):
    """将训练指标写入文件。"""

    def __init__(self, dirpath: Path) -> None:
        self.dirpath = dirpath
        self.dirpath.mkdir(exist_ok=True, parents=True)
        self.train_records: List[dict] = []
        self.valid_records: List[dict] = []

    def on_train_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        self.train_records.append({k: v for k, v in trainer.metrics.items() if not k.startswith("val/")})
        pd.DataFrame.from_records(self.train_records).to_csv(self.dirpath / "train_result.csv", index=True)

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        self.valid_records.append({k: v for k, v in trainer.metrics.items() if k.startswith("val/")})
        pd.DataFrame.from_records(self.valid_records).to_csv(self.dirpath / "validation_result.csv", index=True)


class Checkpoint(Callback):
    """定期保存检查点以实现持久化和恢复。

    参考: https://github.com/PyTorchLightning/pytorch-lightning/blob/bfa8b7be/pytorch_lightning/callbacks/model_checkpoint.py

    参数
    ----------
    dirpath
        保存检查点文件的目录。
    filename
        检查点文件名。可包含命名格式化选项自动填充。
        例如: ``{iter:03d}-{reward:.2f}.pth``。
        支持的参数名有:

        - iter (整数)
        - trainer.metrics中的指标
        - 时间字符串，格式为``%Y%m%d%H%M%S``
    save_latest
        在``latest.pth``中保存最新检查点。
        如果为``link``，``latest.pth``将创建为软链接。
        如果为``copy``，``latest.pth``将保存为独立副本。
        设为none可禁用此功能。
    every_n_iters
        每n次训练迭代结束时保存检查点，
        如果有验证则在验证后保存。
    time_interval
        再次保存检查点的最大时间间隔(秒)。
    save_on_fit_end
        在训练结束时保存最后一个检查点。
        如果该位置已有检查点则不执行任何操作。
    """

    def __init__(
        self,
        dirpath: Path,
        filename: str = "{iter:03d}.pth",
        save_latest: Literal["link", "copy"] | None = "link",
        every_n_iters: int | None = None,
        time_interval: int | None = None,
        save_on_fit_end: bool = True,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.save_latest = save_latest
        self.every_n_iters = every_n_iters
        self.time_interval = time_interval
        self.save_on_fit_end = save_on_fit_end

        self._last_checkpoint_name: str | None = None
        self._last_checkpoint_iter: int | None = None
        self._last_checkpoint_time: float | None = None

    def on_fit_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        if self.save_on_fit_end and (trainer.current_iter != self._last_checkpoint_iter):
            self._save_checkpoint(trainer)

    def on_iter_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        should_save_ckpt = False
        if self.every_n_iters is not None and (trainer.current_iter + 1) % self.every_n_iters == 0:
            should_save_ckpt = True
        if self.time_interval is not None and (
            self._last_checkpoint_time is None or (time.time() - self._last_checkpoint_time) >= self.time_interval
        ):
            should_save_ckpt = True
        if should_save_ckpt:
            self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer: Trainer) -> None:
        self.dirpath.mkdir(exist_ok=True, parents=True)
        self._last_checkpoint_name = self._new_checkpoint_name(trainer)
        self._last_checkpoint_iter = trainer.current_iter
        self._last_checkpoint_time = time.time()
        torch.save(trainer.state_dict(), self.dirpath / self._last_checkpoint_name)

        latest_pth = self.dirpath / "latest.pth"

        # Remove first before saving
        if self.save_latest and (latest_pth.exists() or os.path.islink(latest_pth)):
            latest_pth.unlink()

        if self.save_latest == "link":
            latest_pth.symlink_to(self.dirpath / self._last_checkpoint_name)
        elif self.save_latest == "copy":
            shutil.copyfile(self.dirpath / self._last_checkpoint_name, latest_pth)

    def _new_checkpoint_name(self, trainer: Trainer) -> str:
        return self.filename.format(
            iter=trainer.current_iter, time=datetime.now().strftime("%Y%m%d%H%M%S"), **trainer.metrics
        )
