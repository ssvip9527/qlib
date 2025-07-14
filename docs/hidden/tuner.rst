.. _tuner:

参数调优工具 (Tuner)
=====
.. currentmodule:: qlib

简介
------------

欢迎使用参数调优工具 (Tuner)，本文档假设您已经能够熟练且正确地使用 Estimator。

您可以通过 Tuner 找到模型、训练器、策略和数据标签的最佳超参数及其组合。

`tuner` 程序的使用方法与 `estimator` 类似，您需要提供配置文件的路径。
`tuner` 将执行以下操作：

- 构建调优管道 (tuner pipeline)
- 搜索并保存单个调优器的最佳超参数
- 搜索管道中的下一个调优器
- 保存全局最佳超参数和组合

每个调优器由一种模块组合构成，其目标是搜索该组合的最优超参数。
管道由不同的调优器组成，旨在找到模块的最佳组合。

结果将在屏幕上打印并保存到文件中，您可以在实验保存文件中查看结果。

示例
~~~~~~~

让我们看一个示例：

首先确保您已安装最新版本的 `qlib`。

然后，您需要提供一个配置来设置实验。
我们编写了一个简单的配置示例如下：

.. code-block:: YAML

    experiment:
        name: tuner_experiment
        tuner_class: QLibTuner
    qlib_client:
        auto_mount: False
        logging_level: INFO
    optimization_criteria:
        report_type: model
        report_factor: model_score
        optim_type: max
    tuner_pipeline:
      -
        model:
            class: SomeModel
            space: SomeModelSpace
        trainer:
            class: RollingTrainer
        strategy:
            class: TopkAmountStrategy
            space: TopkAmountStrategySpace
        max_evals: 2

    time_period:
        rolling_period: 360
        train_start_date: 2005-01-01
        train_end_date: 2014-12-31
        validate_start_date: 2015-01-01
        validate_end_date: 2016-06-30
        test_start_date: 2016-07-01
        test_end_date: 2018-04-30
    data:
        class: ALPHA360
        provider_uri: /data/qlib
        args:
            start_date: 2005-01-01
            end_date: 2018-04-30
            dropna_label: True
            dropna_feature: True
        filter:
            market: csi500
            filter_pipeline:
              -
                class: NameDFilter
                module_path: qlib.data.filter
                args:
                  name_rule_re: S(?!Z3)
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
              -
                class: ExpressionDFilter
                module_path: qlib.data.filter
                args:
                  rule_expression: $open/$factor<=45
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
    backtest:
        normal_backtest_args:
            limit_threshold: 0.095
            account: 500000
            benchmark: SH000905
            deal_price: vwap
        long_short_backtest_args:
            topk: 50

接下来，我们运行以下命令，您将看到：

.. code-block:: bash

    ~/v-yindzh/Qlib/cfg$ tuner -c tuner_config.yaml

    Searching params: {'model_space': {'colsample_bytree': 0.8870905643607678, 'lambda_l1': 472.3188735122233, 'lambda_l2': 92.75390994877243, 'learning_rate': 0.09741751430635413, 'loss': 'mse', 'max_depth': 8, 'num_leaves': 160, 'num_threads': 20, 'subsample': 0.7536051584789751}, 'strategy_space': {'buffer_margin': 250, 'topk': 40}}
    ...
    (Estimator experiment screen log)
    ...
    Searching params: {'model_space': {'colsample_bytree': 0.6667379039007301, 'lambda_l1': 382.10698024977904, 'lambda_l2': 117.02506488151757, 'learning_rate': 0.18514539615228137, 'loss': 'mse', 'max_depth': 6, 'num_leaves': 200, 'num_threads': 12, 'subsample': 0.9449255686969292}, 'strategy_space': {'buffer_margin': 200, 'topk': 30}}
    ...
    (Estimator experiment screen log)
    ...
    Local best params: {'model_space': {'colsample_bytree': 0.6667379039007301, 'lambda_l1': 382.10698024977904, 'lambda_l2': 117.02506488151757, 'learning_rate': 0.18514539615228137, 'loss': 'mse', 'max_depth': 6, 'num_leaves': 200, 'num_threads': 12, 'subsample': 0.9449255686969292}, 'strategy_space': {'buffer_margin': 200, 'topk': 30}}
    Time cost: 489.87220 | Finished searching best parameters in Tuner 0.
    Time cost: 0.00069 | Finished saving local best tuner parameters to: tuner_experiment/estimator_experiment/estimator_experiment_0/local_best_params.json .
    Searching params: {'data_label_space': {'labels': ('Ref($vwap, -2)/Ref($vwap, -1) - 2',)}, 'model_space': {'input_dim': 158, 'lr': 0.001, 'lr_decay': 0.9100529502185579, 'lr_decay_steps': 162.48901403763966, 'optimizer': 'gd', 'output_dim': 1}, 'strategy_space': {'buffer_margin': 300, 'topk': 35}}
    ...
    (Estimator experiment screen log)
    ...
    Searching params: {'data_label_space': {'labels': ('Ref($vwap, -2)/Ref($vwap, -1) - 1',)}, 'model_space': {'input_dim': 158, 'lr': 0.1, 'lr_decay': 0.9882802970847494, 'lr_decay_steps': 164.76742865207729, 'optimizer': 'adam', 'output_dim': 1}, 'strategy_space': {'buffer_margin': 250, 'topk': 35}}
    ...
    (Estimator experiment screen log)
    ...
    Local best params: {'data_label_space': {'labels': ('Ref($vwap, -2)/Ref($vwap, -1) - 1',)}, 'model_space': {'input_dim': 158, 'lr': 0.1, 'lr_decay': 0.9882802970847494, 'lr_decay_steps': 164.76742865207729, 'optimizer': 'adam', 'output_dim': 1}, 'strategy_space': {'buffer_margin': 250, 'topk': 35}}
    Time cost: 550.74039 | Finished searching best parameters in Tuner 1.
    Time cost: 0.00023 | Finished saving local best tuner parameters to: tuner_experiment/estimator_experiment/estimator_experiment_1/local_best_params.json .
    Time cost: 1784.14691 | Finished tuner pipeline.
    Time cost: 0.00014 | Finished save global best tuner parameters.
    Best Tuner id: 0.
    You can check the best parameters at tuner_experiment/global_best_params.json.


最后，您可以在指定路径中查看实验结果。

配置文件
------------------

使用 `tuner` 之前，您需要准备一个配置文件。接下来我们将介绍如何准备配置文件的各个部分。

关于实验
~~~~~~~~~~~~~~~~~~~~

首先，您的配置文件需要包含一个关于实验的字段，其键为 `experiment`，该字段及其内容决定保存路径和调优器类。

通常它应包含以下内容：

.. code-block:: YAML

    experiment:
        name: tuner_experiment
        tuner_class: QLibTuner

此外，还有一些可选字段。各字段的含义如下：

- `name`
    实验名称，字符串类型，程序将使用该实验名称创建目录以保存整个实验过程和结果。默认值为 `tuner_experiment`。

- `dir`
    保存路径，字符串类型，程序将在此路径下创建实验目录。默认值为配置文件所在路径。

- `tuner_class`
    调优器类，字符串类型，必须是已实现的模型，例如 `qlib` 中的 `QLibTuner`，或自定义调优器，但必须是 `qlib.contrib.tuner.Tuner` 的子类，默认值为 `QLibTuner`。

- `tuner_module_path`
    模块路径，字符串类型，支持绝对路径，表示调优器实现的路径。默认值为 `qlib.contrib.tuner.tuner`

关于优化标准
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

您需要指定一个优化因子，因为调优器需要一个因子来判断哪个情况比其他情况更好。
通常，我们使用 `estimator` 的结果，例如回测结果和模型分数。

此部分需要包含以下字段：

.. code-block:: YAML

    optimization_criteria:
        report_type: model
        report_factor: model_pearsonr
        optim_type: max

- `report_type`
    报告类型，字符串类型，决定您要使用哪种报告。如果要使用回测结果类型，可选择 `pred_long`、`pred_long_short`、`pred_short`、`excess_return_without_cost` 和 `excess_return_with_cost`。如果要使用模型结果类型，只能选择 `model`。

- `report_factor`
    报告中要使用的因子，字符串类型，决定您要优化哪个因子。如果 `report_type` 是回测结果类型，可选择 `annualized_return`、`information_ratio`、`max_drawdown`、`mean` 和 `std`。如果 `report_type` 是模型结果类型，可选择 `model_score` 和 `model_pearsonr`。

- `optim_type`
    优化类型，字符串类型，决定您要进行哪种优化。您可以最小化或最大化因子，因此可在此字段选择 `max`、`min` 或 `correlation`。
    注意：`correlation` 表示因子的最佳值为 1，例如 `model_pearsonr`（相关系数）。

如果您想处理因子或获取其他类型的因子，可以在自定义调优器中重写 `objective` 方法。

关于调优管道
~~~~~~~~~~~~~~~~~~~~~~~~

The tuner pipeline contains different tuners, and the `tuner` program will process each tuner in pipeline. Each tuner will get an optimal hyper-parameters of its specific combination of modules. The pipeline will contrast the results of each tuner, and get the best combination and its optimal hyper-parameters. So, you need to configurate the pipeline and each tuner, here is an example:

.. code-block:: YAML

    tuner_pipeline:
      -
        model:
            class: SomeModel
            space: SomeModelSpace
        trainer:
            class: RollingTrainer
        strategy:
            class: TopkAmountStrategy
            space: TopkAmountStrategySpace
        max_evals: 2

Each part represents a tuner, and its modules which are to be tuned. Space in each part is the hyper-parameters' space of a certain module, you need to create your searching space and modify it in `/qlib/contrib/tuner/space.py`. We use `hyperopt` package to help us to construct the space, you can see the detail of how to use it in https://github.com/hyperopt/hyperopt/wiki/FMin .

- model
    You need to provide the `class` and the `space` of the model. If the model is user's own implementation, you need to provide the `module_path`.

- trainer
    You need to provide the `class` of the trainer. If the trainer is user's own implementation, you need to provide the `module_path`.

- strategy
    You need to provide the `class` and the `space` of the strategy. If the strategy is user's own implementation, you need to provide the `module_path`.

- data_label
    The label of the data, you can search which kinds of labels will lead to a better result. This part is optional, and you only need to provide `space`.

- max_evals
    Allow up to this many function evaluations in this tuner. The default value is 10.

If you don't want to search some modules, you can fix their spaces in `space.py`. We will not give the default module.

About the time period
~~~~~~~~~~~~~~~~~~~~~

You need to use the same dataset to evaluate your different `estimator` experiments in `tuner` experiment. Two experiments using different dataset are uncomparable. You can specify `time_period` through the configuration file:

.. code-block:: YAML

    time_period:
        rolling_period: 360
        train_start_date: 2005-01-01
        train_end_date: 2014-12-31
        validate_start_date: 2015-01-01
        validate_end_date: 2016-06-30
        test_start_date: 2016-07-01
        test_end_date: 2018-04-30

- `rolling_period`
    The rolling period, integer type, indicates how many time steps need rolling when rolling the data. The default value is `60`. If you use `RollingTrainer`, this config will be used, or it will be ignored.

- `train_start_date`
    Training start time, str type.

- `train_end_date`
    Training end time, str type.

- `validate_start_date`
    Validation start time, str type.

- `validate_end_date`
    Validation end time, str type.

- `test_start_date`
    Test start time, str type.

- `test_end_date`
    Test end time, str type. If `test_end_date` is `-1` or greater than the last date of the data, the last date of the data will be used as `test_end_date`.

About the data and backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`data` and `backtest` are all same in the whole `tuner` experiment. Different `estimator` experiments must use the same data and backtest method. So, these two parts of config are same with that in `estimator` configuration. You can see the precise definition of these parts in `estimator` introduction. We only provide an example here.

.. code-block:: YAML

    data:
        class: ALPHA360
        provider_uri: /data/qlib
        args:
            start_date: 2005-01-01
            end_date: 2018-04-30
            dropna_label: True
            dropna_feature: True
            feature_label_config: /home/v-yindzh/v-yindzh/QLib/cfg/feature_config.yaml
        filter:
            market: csi500
            filter_pipeline:
              -
                class: NameDFilter
                module_path: qlib.filter
                args:
                  name_rule_re: S(?!Z3)
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
              -
                class: ExpressionDFilter
                module_path: qlib.filter
                args:
                  rule_expression: $open/$factor<=45
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
    backtest:
        normal_backtest_args:
            limit_threshold: 0.095
            account: 500000
            benchmark: SH000905
            deal_price: vwap
        long_short_backtest_args:
            topk: 50

Experiment Result
-----------------

All the results are stored in experiment file directly, you can check them directly in the corresponding files.
What we save are as following:

- Global optimal parameters
- Local optimal parameters of each tuner
- Config file of this `tuner` experiment
- Every `estimator` experiments result in the process
