# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt


class HFLGBModel(ModelFT, LightGBMFInt):
    """用于高频预测的LightGBM模型"""

    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:  # 仅支持mse和binary损失函数
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.model = None

    def _cal_signal_metrics(self, y_test, l_cut, r_cut):
        """
        按日级别计算信号指标
        """
        up_pre, down_pre = [], []
        up_alpha_ll, down_alpha_ll = [], []
        for date in y_test.index.get_level_values(0).unique():
            df_res = y_test.loc[date].sort_values("pred")
            if int(l_cut * len(df_res)) < 10:
                warnings.warn("警告：阈值过低或证券数量不足")
                continue
            top = df_res.iloc[: int(l_cut * len(df_res))]
            bottom = df_res.iloc[int(r_cut * len(df_res)) :]

            down_precision = len(top[top[top.columns[0]] < 0]) / (len(top))
            up_precision = len(bottom[bottom[top.columns[0]] > 0]) / (len(bottom))

            down_alpha = top[top.columns[0]].mean()
            up_alpha = bottom[bottom.columns[0]].mean()

            up_pre.append(up_precision)
            down_pre.append(down_precision)
            up_alpha_ll.append(up_alpha)
            down_alpha_ll.append(down_alpha)

        return (
            np.array(up_pre).mean(),
            np.array(down_pre).mean(),
            np.array(up_alpha_ll).mean(),
            np.array(down_alpha_ll).mean(),
        )

    def hf_signal_test(self, dataset: DatasetH, threhold=0.2):
        """
        在高频测试集上测试信号
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        df_test.dropna(inplace=True)
        x_test, y_test = df_test["feature"], df_test["label"]
        # 将标签转换为alpha值
        y_test[y_test.columns[0]] = y_test[y_test.columns[0]] - y_test[y_test.columns[0]].mean(level=0)

        res = pd.Series(self.model.predict(x_test.values), index=x_test.index)
        y_test["pred"] = res

        up_p, down_p, up_a, down_a = self._cal_signal_metrics(y_test, threhold, 1 - threhold)
        print("===============================")
        print("高频信号测试")
        print("===============================")
        print("测试集准确率: ")
        print("正样本准确率: {}, 负样本准确率: {}".format(up_p, down_p))
        print("测试集Alpha平均值: ")
        print("正样本平均alpha: {}, 负样本平均alpha: {}".format(up_a, down_a))

    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("数据集数据为空，请检查您的数据集配置。")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            l_name = df_train["label"].columns[0]
            # 将标签转换为alpha值
            df_train.loc[:, ("label", l_name)] = (
                df_train.loc[:, ("label", l_name)]
                - df_train.loc[:, ("label", l_name)].groupby(level=0, group_keys=False).mean()
            )
            df_valid.loc[:, ("label", l_name)] = (
                df_valid.loc[:, ("label", l_name)]
                - df_valid.loc[:, ("label", l_name)].groupby(level=0, group_keys=False).mean()
            )

            def mapping_fn(x):
                return 0 if x < 0 else 1

            df_train["label_c"] = df_train["label"][l_name].apply(mapping_fn)
            df_valid["label_c"] = df_valid["label"][l_name].apply(mapping_fn)
            x_train, y_train = df_train["feature"], df_train["label_c"].values
            x_valid, y_valid = df_valid["feature"], df_valid["label_c"].values
        else:
            raise ValueError("LightGBM不支持多标签训练")

        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=None,
    ):
        if evals_result is None:
            evals_result = dict()
        dtrain, dvalid = self._prepare_data(dataset)
        early_stopping_callback = lgb.early_stopping(early_stopping_rounds)
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("模型尚未训练！")
        x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
        """
        微调模型

        参数
        ----------
        dataset : DatasetH
            用于微调的数据集
        num_boost_round : int
            微调模型的轮数
        verbose_eval : int
            详细级别
        """
        # Based on existing model and finetune by train more rounds
        dtrain, _ = self._prepare_data(dataset)
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[verbose_eval_callback],
        )
