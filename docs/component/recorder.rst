.. _recorder:

====================================
Qlib 记录器：实验管理
====================================
.. currentmodule:: qlib

简介
============
``Qlib`` 包含一个名为 ``QlibRecorder`` 的实验管理系统，旨在帮助用户高效地处理实验和分析结果。

该系统包含三个组件：

- `ExperimentManager`
    管理实验的类。

- `Experiment`
    实验类，每个实例负责单个实验。

- `Recorder`
    记录器类，每个实例负责单个运行记录。

以下是系统结构的概览：

.. code-block::

    ExperimentManager
        - Experiment 1
            - Recorder 1
            - Recorder 2
            - ...
        - Experiment 2
            - Recorder 1
            - Recorder 2
            - ...
        - ...

此实验管理系统定义了一组接口，并提供了基于机器学习平台 ``MLFlow`` 的具体实现 ``MLflowExpManager`` (`链接 <https://mlflow.org/>`_)。

如果用户将 ``ExpManager`` 的实现设置为 ``MLflowExpManager``，则可以使用命令 `mlflow ui` 可视化和检查实验结果。有关更多信息，请参阅 `此处 <https://www.mlflow.org/docs/latest/cli.html#mlflow-ui>`_ 的相关文档。

Qlib 记录器
=============
``QlibRecorder`` 为用户提供了使用实验管理系统的高级 API。这些接口封装在 ``Qlib`` 中的变量 ``R`` 中，用户可以直接使用 ``R`` 与系统交互。以下命令展示了如何在 Python 中导入 ``R``：

.. code-block:: Python

        from qlib.workflow import R

``QlibRecorder`` 包含多个常用 API，用于在工作流中管理 `实验` 和 `记录器`。有关更多可用 API，请参阅以下关于 `实验管理器`、`实验` 和 `记录器` 的部分。

以下是 ``QlibRecorder`` 的可用接口：

.. autoclass:: qlib.workflow.__init__.QlibRecorder
    :members: 

实验管理器
==================

``Qlib`` 中的 ``ExpManager`` 模块负责管理不同的实验。``ExpManager`` 的大多数 API 与 ``QlibRecorder`` 相似，其中最重要的 API 是 ``get_exp`` 方法。用户可以直接参考上文文档了解如何使用 ``get_exp`` 方法的详细信息。

.. autoclass:: qlib.workflow.expm.ExpManager
    :members: get_exp, list_experiments
    :noindex:

有关其他接口（如 `create_exp`、`delete_exp`），请参阅 `实验管理器 API <../reference/api.html#experiment-manager>`_。

实验
==========

``Experiment`` 类专门负责单个实验，它将处理与实验相关的所有操作。包括 `开始`、`结束` 实验等基本方法。此外，还提供与 `记录器` 相关的方法：如 `get_recorder` 和 `list_recorders`。

.. autoclass:: qlib.workflow.exp.Experiment
    :members: get_recorder, list_recorders
    :noindex:

有关其他接口（如 `search_records`、`delete_recorder`），请参阅 `实验 API <../reference/api.html#experiment>`_。

``Qlib`` 还提供了一个默认的 ``Experiment``，当用户使用 `log_metrics` 或 `get_exp` 等 API 时，会在特定情况下创建和使用该默认实验。如果使用默认 ``Experiment``，运行 ``Qlib`` 时会记录相关信息。用户可以在 ``Qlib`` 的配置文件中或 ``Qlib`` 的 `初始化 <../start/initialization.html#parameters>`_ 过程中更改默认 ``Experiment`` 的名称，默认名称为 '`Experiment`'。

记录器
========

``Recorder`` 类负责单个记录器。它将处理单个运行的详细操作，如 ``log_metrics``、``log_params``。它旨在帮助用户轻松跟踪运行过程中生成的结果和内容

以下是一些未包含在 ``QlibRecorder`` 中的重要 API：

.. autoclass:: qlib.workflow.recorder.Recorder
    :members: list_artifacts, list_metrics, list_params, list_tags
    :noindex:

有关其他接口（如 `save_objects`、`load_object`），请参阅 `记录器 API <../reference/api.html#recorder>`_。

记录模板
===============

``RecordTemp`` 类用于以特定格式生成实验结果，如 IC 和回测结果。我们提供了三种不同的 `记录模板` 类：

- ``SignalRecord``：此类生成模型的 `预测` 结果。
- ``SigAnaRecord``：此类生成模型的 `IC`、`ICIR`、`Rank IC` 和 `Rank ICIR`。

以下是 ``SigAnaRecord`` 中实现的简单示例，用户如果想使用自己的预测和标签计算 IC、Rank IC、多空收益，可以参考此示例。

.. code-block:: Python

    from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

    ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
    long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, 0])

- ``PortAnaRecord``：此类生成 `回测` 结果。有关 `回测` 以及可用 `策略` 的详细信息，用户可以参考 `策略 <../component/strategy.html>`_ 和 `回测 <../component/backtest.html>`_。

以下是 ``PortAnaRecord`` 中实现的简单示例，用户如果想基于自己的预测和标签进行回测，可以参考此示例。

.. code-block:: Python

    from qlib.contrib.strategy.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import (
        backtest as normal_backtest,
        risk_analysis,
    )

    # 回测
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
    }
    BACKTEST_CONFIG = {
        "limit_threshold": 0.095,
        "account": 100000000,
        "benchmark": BENCHMARK,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }

    strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = normal_backtest(pred_score, strategy=strategy,** BACKTEST_CONFIG)

    # 分析
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    print(analysis_df)

有关 API 的更多信息，请参阅 `记录模板 API <../reference/api.html#module-qlib.workflow.record_temp>`_。



已知限制
=================
- Python 对象基于 pickle 保存，当转储对象和加载对象的环境不同时可能会导致问题。
