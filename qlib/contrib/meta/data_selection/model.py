# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm
import copy
from typing import Union, List

from ....model.meta.dataset import MetaTaskDataset
from ....model.meta.model import MetaTaskModel
from ....workflow import R
from .utils import ICLoss
from .dataset import MetaDatasetDS

from qlib.log import get_module_logger
from qlib.model.meta.task import MetaTask
from qlib.data.dataset.weight import Reweighter
from qlib.contrib.meta.data_selection.net import PredNet

logger = get_module_logger("data selection")


class TimeReweighter(Reweighter):
    def __init__(self, time_weight: pd.Series):
        self.time_weight = time_weight

    def reweight(self, data: Union[pd.DataFrame, pd.Series]):
        # TODO: handling TSDataSampler
        w_s = pd.Series(1.0, index=data.index)
        for k, w in self.time_weight.items():
            w_s.loc[slice(*k)] = w
        logger.info(f"重加权结果: {w_s}")
        return w_s


class MetaModelDS(MetaTaskModel):
    """
    基于元学习的数据选择元模型。
    """

    def __init__(
        self,
        step,
        hist_step_n,
        clip_method="tanh",
        clip_weight=2.0,
        criterion="ic_loss",
        lr=0.0001,
        max_epoch=100,
        seed=43,
        alpha=0.0,
        loss_skip_thresh=50,
    ):
        """
        loss_skip_size: int
            每天跳过损失计算的阈值数量。
        """
        self.step = step
        self.hist_step_n = hist_step_n
        self.clip_method = clip_method
        self.clip_weight = clip_weight
        self.criterion = criterion
        self.lr = lr
        self.max_epoch = max_epoch
        self.fitted = False
        self.alpha = alpha
        self.loss_skip_thresh = loss_skip_thresh
        torch.manual_seed(seed)

    def run_epoch(self, phase, task_list, epoch, opt, loss_l, ignore_weight=False):
        if phase == "train":
            self.tn.train()
            torch.set_grad_enabled(True)
        else:
            self.tn.eval()
            torch.set_grad_enabled(False)
        running_loss = 0.0
        pred_y_all = []
        for task in tqdm(task_list, desc=f"{phase} Task", leave=False):
            meta_input = task.get_meta_input()
            pred, weights = self.tn(
                meta_input["X"],
                meta_input["y"],
                meta_input["time_perf"],
                meta_input["time_belong"],
                meta_input["X_test"],
                ignore_weight=ignore_weight,
            )
            if self.criterion == "mse":
                criterion = nn.MSELoss()
                loss = criterion(pred, meta_input["y_test"])
            elif self.criterion == "ic_loss":
                criterion = ICLoss(self.loss_skip_thresh)
                try:
                    loss = criterion(pred, meta_input["y_test"], meta_input["test_idx"])
                except ValueError as e:
                    get_module_logger("MetaModelDS").warning(f"Exception `{e}` when calculating IC loss")
                    continue
            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

            assert not np.isnan(loss.detach().item()), "NaN loss!"

            if phase == "train":
                opt.zero_grad()
                loss.backward()
                opt.step()
            elif phase == "test":
                pass

            pred_y_all.append(
                pd.DataFrame(
                    {
                        "pred": pd.Series(pred.detach().cpu().numpy(), index=meta_input["test_idx"]),
                        "label": pd.Series(meta_input["y_test"].detach().cpu().numpy(), index=meta_input["test_idx"]),
                    }
                )
            )
            running_loss += loss.detach().item()
        running_loss = running_loss / len(task_list)
        loss_l.setdefault(phase, []).append(running_loss)

        pred_y_all = pd.concat(pred_y_all)
        ic = (
            pred_y_all.groupby("datetime", group_keys=False)
            .apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
            .mean()
        )

        R.log_metrics(**{f"loss/{phase}": running_loss, "step": epoch})
        R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})

    def fit(self, meta_dataset: MetaDatasetDS):
        """
        基于元学习的数据选择由于闭式代理测量而直接与元数据集交互。

        参数
        ----------
        meta_dataset : MetaDatasetDS
            元模型将元数据集用于其训练过程。
        """

        if not self.fitted:
            for k in set(["lr", "step", "hist_step_n", "clip_method", "clip_weight", "criterion", "max_epoch"]):
                R.log_params(**{k: getattr(self, k)})

        # FIXME: 获取测试任务仅用于检查性能
        phases = ["train", "test"]
        meta_tasks_l = meta_dataset.prepare_tasks(phases)

        if len(meta_tasks_l[1]):
            R.log_params(
                **dict(proxy_test_begin=meta_tasks_l[1][0].task["dataset"]["kwargs"]["segments"]["test"])
            )  # debug: record when the test phase starts

        self.tn = PredNet(
            step=self.step,
            hist_step_n=self.hist_step_n,
            clip_weight=self.clip_weight,
            clip_method=self.clip_method,
            alpha=self.alpha,
        )

        opt = optim.Adam(self.tn.parameters(), lr=self.lr)

        # run weight with no weight
        for phase, task_list in zip(phases, meta_tasks_l):
            self.run_epoch(f"{phase}_noweight", task_list, 0, opt, {}, ignore_weight=True)
            self.run_epoch(f"{phase}_init", task_list, 0, opt, {})

        # run training
        loss_l = {}
        for epoch in tqdm(range(self.max_epoch), desc="epoch"):
            for phase, task_list in zip(phases, meta_tasks_l):
                self.run_epoch(phase, task_list, epoch, opt, loss_l)
            R.save_objects(**{"model.pkl": self.tn})
        self.fitted = True

    def _prepare_task(self, task: MetaTask) -> dict:
        meta_ipt = task.get_meta_input()
        weights = self.tn.twm(meta_ipt["time_perf"])

        weight_s = pd.Series(weights.detach().cpu().numpy(), index=task.meta_info.columns)
        task = copy.copy(task.task)  # 注意：这是一个浅拷贝。
        task["reweighter"] = TimeReweighter(weight_s)
        return task

    def inference(self, meta_dataset: MetaTaskDataset) -> List[dict]:
        res = []
        for mt in meta_dataset.prepare_tasks("test"):
            res.append(self._prepare_task(mt))
        return res
