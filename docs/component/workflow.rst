.. _workflow:

=============================
工作流：工作流管理
=============================
.. currentmodule:: qlib

简介
============

`Qlib框架 <../introduction/introduction.html#framework>`_ 中的组件采用松耦合设计。用户可以使用这些组件构建自己的量化研究工作流，例如`示例 <https://github.com/ssvip9527/qlib/blob/main-cn/examples/workflow_by_code.py>`_。


此外，``Qlib``提供了更友好的接口``qrun``来自动运行通过配置定义的整个工作流。运行整个工作流称为一次`执行`。
使用``qrun``，用户可以轻松启动一次`执行`，包括以下步骤：

- Data
    - Loading
    - Processing
    - Slicing
- Model
    - Training and inference
    - Saving & loading
- Evaluation
    - Forecast signal analysis
    - Backtest

对于每次`执行`，``Qlib``有一个完整的系统来跟踪训练、推理和评估阶段生成的所有信息和工件。有关``Qlib``如何处理此问题的更多信息，请参考相关文档：`Recorder：实验管理 <../component/recorder.html>`_。

完整示例
================

在深入细节之前，这里有一个``qrun``的完整示例，定义了典型量化研究中的工作流。
以下是``qrun``的典型配置文件。

.. code-block:: YAML

    qlib_init:
        provider_uri: "~/.qlib/qlib_data/cn_data"
        region: cn
    market: &market csi300
    benchmark: &benchmark SH000300
    data_handler_config: &data_handler_config
        start_time: 2008-01-01
        end_time: 2020-08-01
        fit_start_time: 2008-01-01
        fit_end_time: 2014-12-31
        instruments: *market
    port_analysis_config: &port_analysis_config
        strategy:
            class: TopkDropoutStrategy
            module_path: qlib.contrib.strategy.strategy
            kwargs:
                topk: 50
                n_drop: 5
                signal: <PRED>
        backtest:
            start_time: 2017-01-01
            end_time: 2020-08-01
            account: 100000000
            benchmark: *benchmark
            exchange_kwargs:
                limit_threshold: 0.095
                deal_price: close
                open_cost: 0.0005
                close_cost: 0.0015
                min_cost: 5
    task:
        model:
            class: LGBModel
            module_path: qlib.contrib.model.gbdt
            kwargs:
                loss: mse
                colsample_bytree: 0.8879
                learning_rate: 0.0421
                subsample: 0.8789
                lambda_l1: 205.6999
                lambda_l2: 580.9768
                max_depth: 8
                num_leaves: 210
                num_threads: 20
        dataset:
            class: DatasetH
            module_path: qlib.data.dataset
            kwargs:
                handler:
                    class: Alpha158
                    module_path: qlib.contrib.data.handler
                    kwargs: *data_handler_config
                segments:
                    train: [2008-01-01, 2014-12-31]
                    valid: [2015-01-01, 2016-12-31]
                    test: [2017-01-01, 2020-08-01]
        record:
            - class: SignalRecord
              module_path: qlib.workflow.record_temp
              kwargs: {}
            - class: PortAnaRecord
              module_path: qlib.workflow.record_temp
              kwargs:
                  config: *port_analysis_config

将配置保存到`configuration.yaml`后，用户可以通过以下单个命令启动工作流并测试他们的想法。

.. code-block:: bash

    qrun configuration.yaml

如果用户想在调试模式下使用``qrun``，请使用以下命令：

.. code-block:: bash

    python -m pdb qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

.. note::

    安装``Qlib``时，`qrun`将被放置在您的$PATH目录中。

.. note::

    `yaml`文件中的符号`&`表示字段的锚点，当其他字段将此参数作为值的一部分包含时非常有用。以上述配置文件为例，用户可以直接更改`market`和`benchmark`的值，而无需遍历整个配置文件。


配置文件
==================

在本节中，我们将详细介绍``qrun``。
使用``qrun``之前，用户需要准备一个配置文件。以下内容展示了如何准备配置文件的各个部分。

配置文件的设计逻辑非常简单。它预定义了固定的工作流，并提供此yaml接口让用户定义如何初始化每个组件。
它遵循`init_instance_by_config <https://github.com/ssvip9527/qlib/blob/2aee9e0145decc3e71def70909639b5e5a6f4b58/qlib/utils/__init__.py#L264>`_ 的设计。它定义了Qlib每个组件的初始化，通常包括类和初始化参数。

例如，以下yaml和代码是等效的。

.. code-block:: YAML

    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.0421
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20


.. code-block:: python

        from qlib.contrib.model.gbdt import LGBModel
        kwargs = {
            "loss": "mse" ,
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        }
        LGBModel(kwargs)


Qlib初始化部分
-----------------

首先，配置文件需要包含几个用于qlib初始化的基本参数。

.. code-block:: YAML

    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

每个字段的含义如下：

- `provider_uri`
    类型：str。Qlib数据的URI。例如，它可以是``get_data.py``加载的数据存储的位置。

- `region`
    - 如果`region` == "us"，``Qlib``将在美股模式下初始化。
    - 如果`region` == "cn"，``Qlib``将在A股模式下初始化。

    .. note::

        `region`的值应与`provider_uri`中存储的数据一致。


任务部分
------------

配置中的`task`字段对应一个`任务`，包含三个不同子部分的参数：`Model`（模型）、`Dataset`（数据集）和`Record`（记录）。

模型部分
~~~~~~~~~~~~~

In the `task` field, the `model` section describes the parameters of the model to be used for training and inference. For more information about the base ``Model`` class, please refer to `Qlib Model <../component/model.html>`_.

.. code-block:: YAML

    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.0421
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20

每个字段的含义如下：

- `class`
    类型：str。模型类的名称。

- `module_path`
    类型：str。模型在qlib中的路径。

- `kwargs`
    模型的关键字参数。有关更多信息，请参考具体的模型实现：`models <https://github.com/ssvip9527/qlib/blob/main-cn/qlib/contrib/model>`_。

.. note::

    ``Qlib``提供了一个名为``init_instance_by_config``的工具，用于初始化``Qlib``内部的任何类，配置包括字段：`class`、`module_path`和`kwargs`。

数据集部分
~~~~~~~~~~~~~~~

`dataset`字段描述了``Qlib``中``Dataset``模块以及``DataHandler``模块的参数。有关``Dataset``模块的更多信息，请参考`Qlib数据 <../component/data.html#dataset>`_。

``DataHandler``的关键字参数配置如下：

.. code-block:: YAML

    data_handler_config: &data_handler_config
        start_time: 2008-01-01
        end_time: 2020-08-01
        fit_start_time: 2008-01-01
        fit_end_time: 2014-12-31
        instruments: *market

用户可以参考`DataHandler <../component/data.html#datahandler>`_文档了解配置中每个字段的含义。

以下是``Dataset``模块的配置，负责训练和测试阶段的数据预处理和切片。

.. code-block:: YAML

    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]

记录部分
~~~~~~~~~~~~~~

`record`字段涉及``Qlib``中``Record``模块的参数。``Record``负责以标准格式跟踪训练过程和结果，如`信息系数（IC）`和`回测`。

以下是`回测`的配置以及`回测`中使用的`策略`：

.. code-block:: YAML

    port_analysis_config: &port_analysis_config
        strategy:
            class: TopkDropoutStrategy
            module_path: qlib.contrib.strategy.strategy
            kwargs:
                topk: 50
                n_drop: 5
                signal: <PRED>
        backtest:
            limit_threshold: 0.095
            account: 100000000
            benchmark: *benchmark
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

有关`strategy`和`backtest`配置中每个字段的含义的更多信息，用户可以查阅文档：`策略 <../component/strategy.html>`_和`回测 <../component/backtest.html>`_。

以下是不同`记录模板`（如``SignalRecord``和``PortAnaRecord``）的配置详情：

.. code-block:: YAML

    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs: {}
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config

有关``Qlib``中``Record``模块的更多信息，用户可以参考相关文档：`记录 <../component/recorder.html#record-template>`_。
