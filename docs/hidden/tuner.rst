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

调参管道包含不同的调参器，`tuner`程序将按顺序处理管道中的每个调参器。每个调参器会为其特定的模块组合找到最优超参数。管道会对比各个调参器的结果，从而获得最佳组合及其最优超参数。因此，您需要配置管道和每个调参器，以下是示例：

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

每个部分代表一个调参器及其需要调优的模块。每个部分中的Space是特定模块的超参数空间，您需要创建自己的搜索空间并在`/qlib/contrib/tuner/space.py`中修改。我们使用`hyperopt`包来帮助构建空间，您可以在https://github.com/hyperopt/hyperopt/wiki/FMin查看详细使用方法。

- model
    您需要提供模型的`class`和`space`。如果模型是用户自己实现的，还需要提供`module_path`。

- trainer
    您需要提供训练器的`class`。如果训练器是用户自己实现的，还需要提供`module_path`。

- strategy
    您需要提供策略的`class`和`space`。如果策略是用户自己实现的，还需要提供`module_path`。

- data_label
    数据的标签，您可以搜索哪种标签能带来更好的结果。这部分是可选的，只需提供`space`。

- max_evals
    允许此调参器进行的函数评估次数上限。默认值为10。

如果您不想搜索某些模块，可以在`space.py`中固定它们的空间。我们不会提供默认模块。

关于时间周期
~~~~~~~~~~~~~~~~~~~~~

在`tuner`实验中，您需要使用相同的数据集来评估不同的`estimator`实验。使用不同数据集的两个实验是不可比的。您可以通过配置文件指定`time_period`：

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
    滚动周期，整数类型，表示滚动数据时需要滚动的时间步数。默认值为`60`。如果使用`RollingTrainer`，将使用此配置，否则将被忽略。

- `train_start_date`
    训练开始时间，字符串类型。

- `train_end_date`
    训练结束时间，字符串类型。

- `validate_start_date`
    验证开始时间，字符串类型。

- `validate_end_date`
    验证结束时间，字符串类型。

- `test_start_date`
    测试开始时间，字符串类型。

- `test_end_date`
    测试结束时间，字符串类型。如果`test_end_date`为`-1`或大于数据的最后日期，则使用数据的最后日期作为`test_end_date`。

关于数据和回测
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在整个`tuner`实验中，`data`和`backtest`都是相同的。不同的`estimator`实验必须使用相同的数据和回测方法。因此，这两部分配置与`estimator`配置中的相同。您可以在`estimator`介绍中查看这些部分的精确定义。这里仅提供示例。

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

实验结果
-----------------

所有结果直接存储在实验文件中，您可以直接在相应文件中查看。
我们保存的内容如下：

- 全局最优参数
- 每个调参器的局部最优参数
- 此`tuner`实验的配置文件
- 过程中每个`estimator`实验的结果
