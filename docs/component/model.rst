.. _model:

===========================================
预测模型：模型训练与预测
===========================================

介绍
============

``预测模型（Forecast Model）``旨在生成股票的`预测分数`。用户可以通过``qrun``在自动化工作流中使用``预测模型``，详情请参考`工作流：工作流管理 <workflow.html>`_。

由于``Qlib``中的组件采用松耦合设计，``预测模型``也可以作为独立模块使用。

基类与接口
======================

``Qlib``提供了一个基类`qlib.model.base.Model <../reference/api.html#module-qlib.model.base>`_，所有模型都应继承自该基类。

基类提供以下接口：

.. autoclass:: qlib.model.base.Model
    :members:
    :noindex:

``Qlib``还提供了一个基类`qlib.model.base.ModelFT <../reference/api.html#qlib.model.base.ModelFT>`_，包含模型微调的方法。

关于其他接口如`finetune`，请参考`模型API <../reference/api.html#module-qlib.model.base>`_。

示例
=======

``Qlib``的`模型库（Model Zoo）`包含``LightGBM``、``MLP``、``LSTM``等模型。这些模型被视为``预测模型``的基准。以下步骤展示如何将``LightGBM``作为独立模块运行。

- 首先使用`qlib.init`初始化``Qlib``，详情请参考`初始化 <../start/initialization.html>`_。
- 运行以下代码以获得 `预测分数` `pred_score`
    .. code-block:: Python

        from qlib.contrib.model.gbdt import LGBModel
        from qlib.contrib.data.handler import Alpha158
        from qlib.utils import init_instance_by_config, flatten_dict
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

        market = "csi300"
        benchmark = "SH000300"

        data_handler_config = {
            "start_time": "2008-01-01",
            "end_time": "2020-08-01",
            "fit_start_time": "2008-01-01",
            "fit_end_time": "2014-12-31",
            "instruments": market,
        }

        task = {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                },
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "train": ("2008-01-01", "2014-12-31"),
                        "valid": ("2015-01-01", "2016-12-31"),
                        "test": ("2017-01-01", "2020-08-01"),
                    },
                },
            },
        }

        # 模型初始化
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])

        # 开始实验
        with R.start(experiment_name="workflow"):
            # 训练
            R.log_params(**flatten_dict(task))
            model.fit(dataset)

            # 预测
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

    .. note::

        `Alpha158`是``Qlib``提供的数据处理器，详情请参考`数据处理器 <data.html#data-handler>`_。
        `SignalRecord`是``Qlib``中的`记录模板`，详情请参考`工作流 <recorder.html#record-template>`_。

此外，上述示例已在``examples/train_backtest_analyze.ipynb``中给出。
从技术上讲，模型预测的含义取决于用户设计的标签设置。
默认情况下，分数通常表示预测模型对工具的评级。分数越高，工具的收益越高。


自定义模型
============

Qlib支持自定义模型。如果用户有兴趣定制自己的模型并将其集成到``Qlib``中，请参考`自定义模型集成 <../start/integration.html>`_。


API
===
请参考`模型API <../reference/api.html#module-qlib.model.base>`_。
