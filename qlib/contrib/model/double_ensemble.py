# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...log import get_module_logger


class DEnsembleModel(Model, FeatureInt):
    """双集成模型"""

    def __init__(
        self,
        base_model="gbm",
        loss="mse",
        num_models=6,
        enable_sr=True,
        enable_fs=True,
        alpha1=1.0,
        alpha2=1.0,
        bins_sr=10,
        bins_fs=5,
        decay=None,
        sample_ratios=None,
        sub_weights=None,
        epochs=100,
        early_stopping_rounds=None,
        **kwargs,
    ):
        self.base_model = base_model  # "gbm"或"mlp"，具体地，我们使用lgbm作为"gbm"
        self.num_models = num_models  # 子模型数量
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay
        if sample_ratios is None:  # sample_ratios的默认值
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:  # sub_weights的默认值
            sub_weights = [1] * self.num_models
        if not len(sample_ratios) == bins_fs:
            raise ValueError("sample_ratios的长度应等于bins_fs。")
        self.sample_ratios = sample_ratios
        if not len(sub_weights) == num_models:
            raise ValueError("sub_weights的长度应等于num_models。")
        self.sub_weights = sub_weights
        self.epochs = epochs
        self.logger = get_module_logger("DEnsembleModel")
        self.logger.info("双集成模型...")
        self.ensemble = []  # 当前集成模型，包含所有子模型的列表
        self.sub_features = []  # 每个子模型的特征，格式为pandas.Index
        self.params = {"objective": loss}
        self.params.update(kwargs)
        self.loss = loss
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("数据集数据为空，请检查您的数据集配置。")
        x_train, y_train = df_train["feature"], df_train["label"]
        # 初始化样本权重
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        # 初始化特征
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)
        # 训练子模型
        for k in range(self.num_models):
            self.sub_features.append(features)
            self.logger.info("训练子模型：({}/{}) ".format(k + 1, self.num_models))
            model_k = self.train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)
            # 最后一个子模型不需要进一步的样本重加权和特征选择
            if k + 1 == self.num_models:
                break

            self.logger.info("获取损失曲线和损失值...")
            loss_curve = self.retrieve_loss_curve(model_k, df_train, features)
            pred_k = self.predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k
            pred_ensemble = (pred_sub.iloc[:, : k + 1] * self.sub_weights[0 : k + 1]).sum(axis=1) / np.sum(
                self.sub_weights[0 : k + 1]
            )
            loss_values = pd.Series(self.get_loss(y_train.values.squeeze(), pred_ensemble.values))

            if self.enable_sr:
                self.logger.info("样本重加权...")
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                self.logger.info("特征选择...")
                features = self.feature_selection(df_train, loss_values)

    def train_submodel(self, df_train, df_valid, weights, features):
        dtrain, dvalid = self._prepare_data_gbm(df_train, df_valid, weights, features)
        evals_result = dict()

        callbacks = [lgb.log_evaluation(20), lgb.record_evaluation(evals_result)]
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            self.logger.info("使用早停法训练...")

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.epochs,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]
        return model

    def _prepare_data_gbm(self, df_train, df_valid, weights, features):
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"]

        # LightGBM需要一维数组作为标签
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM不支持多标签训练")

        dtrain = lgb.Dataset(x_train, label=y_train, weight=weights)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def sample_reweight(self, loss_curve, loss_values, k_th):
        """
        双集成的样本重加权（SR）模块
        :param loss_curve: 形状为NxT
        前一个子模型的损失曲线，其中元素(i, t)是在前一个子模型训练的第t次迭代后，第i个样本的误差。
        :param loss_values: 形状为N
        当前集成模型在第i个样本上的损失。
        :param k_th: 当前子模型的索引，从1开始
        :return: weights
        所有样本的权重。
        """
        # 使用排序归一化损失曲线和损失值
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)

        # 从损失曲线计算l_start和l_end
        N, T = loss_curve.shape
        part = np.maximum(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # 计算每个样本的h值
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # 计算权重
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)
        return weights

    def feature_selection(self, df_train, loss_values):
        """
        双集成的特征选择（FS）模块
        :param df_train: 形状为NxF
        :param loss_values: 形状为N
        当前集成模型在第i个样本上的损失。
        :return: res_feat: pandas.Index格式

        """
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)

        # 打乱特定列并计算每个特征的g值
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_train_tmp.loc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), index=x_train_tmp.index
                    )
                    / M
                )
            loss_feat = self.get_loss(y_train.values.squeeze(), pred.values)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
            x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()

        # 训练特征中的某一列全为NaN # 如果g['g_value']存在缺失值
        g["g_value"].replace(np.nan, 0, inplace=True)

        # 将特征分为bins_fs个区间
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)

        # 从区间中随机采样特征以构建新特征集
        res_feat = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
        return pd.Index(set(res_feat))

    def get_loss(self, label, pred):
        if self.loss == "mse":
            return (label - pred) ** 2
        else:
            raise ValueError("尚未实现")

    def retrieve_loss_curve(self, model, df_train, features):
        if self.base_model == "gbm":
            num_trees = model.num_trees()
            x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
            # Lightgbm需要一维数组作为标签
            if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                y_train = np.squeeze(y_train.values)
            else:
                raise ValueError("LightGBM不支持多标签训练")

            N = x_train.shape[0]
            loss_curve = pd.DataFrame(np.zeros((N, num_trees)))
            pred_tree = np.zeros(N, dtype=float)
            for i_tree in range(num_trees):
                pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
                loss_curve.iloc[:, i_tree] = self.get_loss(y_train, pred_tree)
        else:
            raise ValueError("尚未实现")
        return loss_curve

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.ensemble is None:
            raise ValueError("模型尚未训练！")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            pred += (
                pd.Series(submodel.predict(x_test.loc[:, feat_sub].values), index=x_test.index)
                * self.sub_weights[i_sub]
            )
        pred = pred / np.sum(self.sub_weights)
        return pred

    def predict_sub(self, submodel, df_data, features):
        x_data = df_data["feature"].loc[:, features]
        pred_sub = pd.Series(submodel.predict(x_data.values), index=x_data.index)
        return pred_sub

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """获取特征重要性

        注意
        -----
            参数参考：
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        res = []
        for _model, _weight in zip(self.ensemble, self.sub_weights):
            res.append(pd.Series(_model.feature_importance(*args, **kwargs), index=_model.feature_name()) * _weight)
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)
