========================
自定义模型集成
========================

简介
============

``Qlib`` 的 `模型库` 包含 ``LightGBM``、``MLP``、``LSTM`` 等模型。这些模型是 ``预测模型`` 的示例。除了 ``Qlib`` 提供的默认模型外，用户还可以将自己的自定义模型集成到 ``Qlib`` 中。

用户可以按照以下步骤集成自己的自定义模型。

- 定义一个自定义模型类，该类应该是 `qlib.model.base.Model <../reference/api.html#module-qlib.model.base>`_ 的子类。
- 编写一个配置文件，描述自定义模型的路径和参数。
- 测试自定义模型。

自定义模型类
==================
自定义模型需要继承 `qlib.model.base.Model <../reference/api.html#module-qlib.model.base>`_ 并重写其中的方法。

- 重写 `__init__` 方法
    - ``Qlib`` 将初始化参数传递给 \_\_init\_\_ 方法。
    - 配置文件中模型的超参数必须与 `__init__` 方法中定义的参数一致。
    - 代码示例：在以下示例中，配置文件中模型的超参数应包含 `loss:mse` 等参数。

        .. code-block:: Python

            def __init__(self, loss='mse', **kwargs):
                if loss not in {'mse', 'binary'}:
                    raise NotImplementedError
                self._scorer = mean_squared_error if loss == 'mse' else roc_auc_score
                self._params.update(objective=loss, **kwargs)
                self._model = None

- 重写 `fit` 方法
    - ``Qlib`` 调用 fit 方法来训练模型。
    - 参数必须包含训练特征 `dataset`，这是接口中设计的。
    - 参数可以包含一些带有默认值的 `可选` 参数，例如 `GBDT` 的 `num_boost_round = 1000`。
    - 代码示例：在以下示例中，`num_boost_round = 1000` 是一个可选参数。

        .. code-block:: Python

            def fit(self, dataset: DatasetH, num_boost_round = 1000, **kwargs):

                # 为 lgb 训练和评估准备数据集
                df_train, df_valid = dataset.prepare(
                    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                x_train, y_train = df_train["feature"], df_train["label"]
                x_valid, y_valid = df_valid["feature"], df_valid["label"]

                # Lightgbm 需要 1D 数组作为标签
                if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                    y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
                else:
                    raise ValueError("LightGBM 不支持多标签训练")

                dtrain = lgb.Dataset(x_train.values, label=y_train)
                dvalid = lgb.Dataset(x_valid.values, label=y_valid)

                # 拟合模型
                self.model = lgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    valid_sets=[dtrain, dvalid],
                    valid_names=["train", "valid"],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                    evals_result=evals_result,
                    **kwargs
                )

- 重写 `predict` 方法
    - 参数必须包含 `dataset` 参数，该参数将用于获取测试数据集。
    - 返回 `预测分数`。
    - 请参考 `模型 API <../reference/api.html#module-qlib.model.base>`_ 了解 fit 方法的参数类型。
    - 代码示例：在以下示例中，用户需要使用 `LightGBM` 预测测试数据 `x_test` 的标签（例如 `preds`）并返回它。

        .. code-block:: Python

            def predict(self, dataset: DatasetH, **kwargs)-> pandas.Series:
                if self.model is None:
                    raise ValueError("model is not fitted yet!")
                x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
                return pd.Series(self.model.predict(x_test.values), index=x_test.index)

- 重写 `finetune` 方法（可选）
    - 此方法对用户来说是可选的。当用户想在自己的模型上使用此方法时，他们应该继承 ``ModelFT`` 基类，该类包含 `finetune` 接口。
    - 参数必须包含 `dataset` 参数。
    - 代码示例：在以下示例中，用户将使用 `LightGBM` 作为模型并对其进行微调。

        .. code-block:: Python

            def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
                # 基于已有模型继续训练更多轮次进行微调
                dtrain, _ = self._prepare_data(dataset)
                self.model = lgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    init_model=self.model,
                    valid_sets=[dtrain],
                    valid_names=["train"],
                    verbose_eval=verbose_eval,
                )

配置文件
==================

配置文件在 `工作流 <../component/workflow.html#complete-example>`_ 文档中有详细描述。为了将自定义模型集成到 ``Qlib`` 中，用户需要修改配置文件中的 “model” 字段。该配置描述了要使用的模型以及如何初始化它。

- 示例：以下示例描述了上述自定义 LightGBM 模型的配置文件中的 `model` 字段，其中 `module_path` 是模块路径，`class` 是类名，`args` 是传递给 `__init__` 方法的超参数。除 `loss = mse` 外，该字段中的所有参数都通过 `**kwargs` 传递给 `self._params`。

    .. code-block:: YAML

        model:
            class: LGBModel
            module_path: qlib.contrib.model.gbdt
            args:
                loss: mse
                colsample_bytree: 0.8879
                learning_rate: 0.0421
                subsample: 0.8789
                lambda_l1: 205.6999
                lambda_l2: 580.9768
                max_depth: 8
                num_leaves: 210
                num_threads: 20

用户可以在 ``examples/benchmarks`` 中找到 ``模型`` 基线的配置文件。不同模型的所有配置都列在相应的模型文件夹下。

模型测试
=============
假设配置文件为 ``examples/benchmarks/LightGBM/workflow_config_lightgbm.yaml``，用户可以运行以下命令来测试自定义模型：

.. code-block:: bash

    cd examples  # 避免在包含`qlib`的目录下运行程序
    qrun benchmarks/LightGBM/workflow_config_lightgbm.yaml

.. note:: ``qrun`` 是 ``Qlib`` 的内置命令。

此外，``模型`` 也可以作为单个模块进行测试。``examples/workflow_by_code.ipynb`` 中给出了一个示例。

参考资料
=========
要了解有关 ``预测模型`` 的更多信息，请参考 `预测模型：模型训练与预测 <../component/model.html>`_ 和 `模型 API <../reference/api.html#module-qlib.model.base>`_。
