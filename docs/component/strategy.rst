.. _strategy:

========================================
投资组合策略：投资组合管理
========================================
.. currentmodule:: qlib

简介
============

``投资组合策略``旨在采用不同的投资组合策略，即用户可以基于``预测模型``的预测分数，采用不同算法生成投资组合。用户可以通过``工作流``模块在自动化工作流中使用``投资组合策略``，详情请参考`工作流：工作流管理 <workflow.html>`_。

由于``Qlib``中的组件采用松耦合设计，``投资组合策略``也可以作为独立模块使用。

``Qlib``提供了多种已实现的投资组合策略。同时，``Qlib``支持自定义策略，用户可以根据自身需求定制策略。

用户指定模型（预测信号）和策略后，运行回测可以帮助用户检查自定义模型（预测信号）/策略的性能。

基类与接口
======================

BaseStrategy
------------

Qlib提供了一个基类 ``qlib.strategy.base.BaseStrategy``。所有策略类都需要继承该基类并实现其接口。

- `generate_trade_decision`
    generate_trade_decision是一个关键接口，用于在每个交易周期生成交易决策。
    调用此方法的频率取决于执行器频率（默认"time_per_step"="day"）。但交易频率可由用户实现决定。
    例如，如果用户希望以周为单位进行交易，而执行器中的`time_per_step`为"day"，用户可以每周返回非空的TradeDecision（否则返回空值，如`此处 <https://github.com/microsoft/qlib/blob/main/qlib/contrib/strategy/signal_strategy.py#L132>`_ 所示）。

用户可以继承`BaseStrategy`来自定义自己的策略类。

WeightStrategyBase
------------------

Qlib还提供了一个类 ``qlib.contrib.strategy.WeightStrategyBase``，它是`BaseStrategy`的子类。

`WeightStrategyBase`仅关注目标仓位，并根据仓位自动生成订单列表。它提供了`generate_target_weight_position`接口。

- `generate_target_weight_position`
    - 根据当前仓位和交易日期生成目标仓位。输出的权重分布中不考虑现金。
    - 返回目标仓位。

    .. note::
        这里的`target position`指的是总资产的目标百分比。

`WeightStrategyBase`实现了`generate_order_list`接口，其流程如下：

- 调用`generate_target_weight_position`方法生成目标仓位。
- 根据目标仓位生成股票的目标数量。
- 根据目标数量生成订单列表。

用户可以继承`WeightStrategyBase`并实现`generate_target_weight_position`接口来自定义仅关注目标仓位的策略类。

已实现的策略
====================

Qlib provides a implemented strategy classes named `TopkDropoutStrategy`.

TopkDropoutStrategy
-------------------
`TopkDropoutStrategy`是`BaseStrategy`的子类，并实现了`generate_order_list`接口，其流程如下：

- 采用``Topk-Drop``算法计算每只股票的目标数量

    .. note::
        ``Topk-Drop``算法有两个参数：

        - `Topk`：持有的股票数量
        - `Drop`：每个交易日卖出的股票数量

        一般来说，当前持有的股票数量为`Topk`，但在交易初期可能为零。
        每个交易日，将当前持有的证券按预测分数从高到低排序，令$d$为排名$\gt K$的证券数量。
        然后，卖出当前持有的`d`只预测分数最差的股票，并买入相同数量的未持有的预测分数最好的股票。

        通常情况下，$d=$`Drop`，特别是当候选证券池较大、$K$较大且`Drop`较小时。

        在大多数情况下，``TopkDrop``算法每个交易日卖出和买入`Drop`只股票，换手率为2$\times$`Drop`/$K$。

        下图展示了一个典型场景：

        .. image:: ../_static/img/topk_drop.png
            :alt: Topk-Drop



- 根据目标数量生成订单列表

EnhancedIndexingStrategy
------------------------
`EnhancedIndexingStrategy`（增强型指数策略）结合了主动管理和被动管理的特点，
旨在在控制风险暴露（也称为跟踪误差）的同时，使投资组合收益优于基准指数（如标准普尔500指数）。

有关更多信息，请参考`qlib.contrib.strategy.signal_strategy.EnhancedIndexingStrategy`
和`qlib.contrib.strategy.optimizer.enhanced_indexing.EnhancedIndexingOptimizer`。


使用方法与示例
===============

First, user can create a model to get trading signals(the variable name is ``pred_score`` in following cases).

预测分数
----------------

`prediction score`（预测分数）是一个pandas DataFrame。其索引为<datetime(pd.Timestamp), instrument(str)>，并且必须包含一个`score`列。

预测样本如下所示：

.. code-block:: python

      datetime instrument     score
    2019-01-04   SH600000 -0.505488
    2019-01-04   SZ002531 -0.320391
    2019-01-04   SZ000999  0.583808
    2019-01-04   SZ300569  0.819628
    2019-01-04   SZ001696 -0.137140
                 ...            ...
    2019-04-30   SZ000996 -1.027618
    2019-04-30   SH603127  0.225677
    2019-04-30   SH603126  0.462443
    2019-04-30   SH603133 -0.302460
    2019-04-30   SZ300760 -0.126383

``Forecast Model``（预测模型）模块可以进行预测，详情请参考`预测模型：模型训练与预测 <model.html>`_。

通常，预测分数是模型的输出。但有些模型是从不同尺度的标签中学习的，因此预测分数的尺度可能与您的期望不同（例如证券的回报率）。

Qlib没有添加将预测分数缩放到统一尺度的步骤，原因如下：
- 并非所有交易策略都关心尺度（例如TopkDropoutStrategy只关心顺序）。因此，策略负责重新缩放预测分数（例如某些基于投资组合优化的策略可能需要有意义的尺度）。
- 模型可以灵活定义目标、损失和数据处理。因此，我们认为仅基于模型输出直接将其缩放到有意义的值并没有万能的方法。如果您想将其缩放到有意义的值（例如股票回报率），一个直观的解决方案是为模型最近的输出和您最近的目标值创建一个回归模型。

运行回测
----------------

- 在大多数情况下，用户可以使用``backtest_daily``对其投资组合管理策略进行回测。

    .. code-block:: python

        from pprint import pprint

        import qlib
        import pandas as pd
        from qlib.utils.time import Freq
        from qlib.utils import flatten_dict
        from qlib.contrib.evaluate import backtest_daily
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        # 初始化qlib
        qlib.init(provider_uri=<qlib数据目录>)

        CSI300_BENCH = "SH000300"
        STRATEGY_CONFIG = {
            "topk": 50,
            "n_drop": 5,
            # pred_score, pd.Series
            "signal": pred_score,
        }


        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time="2017-01-01", end_time="2020-08-01", strategy=strategy_obj
        )
        analysis = dict()
        # 默认频率为每日（即"day"）
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)



- 如果用户希望更详细地控制其策略（例如用户有更高级版本的执行器），可以按照以下示例操作：

    .. code-block:: python

        from pprint import pprint

        import qlib
        import pandas as pd
        from qlib.utils.time import Freq
        from qlib.utils import flatten_dict
        from qlib.backtest import backtest, executor
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        # 初始化qlib
        qlib.init(provider_uri=<qlib数据目录>)

        CSI300_BENCH = "SH000300"
        # 基准用于计算策略的超额收益。
        # 其数据格式类似于**一个普通的证券**。
        # 例如，您可以使用以下代码查询其数据
        # `D.features(["SH000300"], ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`
        # 它与参数`market`不同，`market`表示股票的集合（例如**一组**像沪深300这样的股票）
        # 例如，您可以使用以下代码查询股票市场的所有数据：
        # ` D.features(D.instruments(market='csi300'), ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`

        FREQ = "day"
        STRATEGY_CONFIG = {
            "topk": 50,
            "n_drop": 5,
            # pred_score, pd.Series
            "signal": pred_score,
        }

        EXECUTOR_CONFIG = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }

        backtest_config = {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": FREQ,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }

        # 策略对象
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        # 执行器对象
        executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
        # 回测
        portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
        # 回测信息
        report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

        # 分析
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"], freq=analysis_freq
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        # 记录指标
        analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
        # 打印结果
        pprint(f"以下是基准收益的分析结果({analysis_freq})。")
        pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
        pprint(f"以下是无成本超额收益的分析结果({analysis_freq})。")
        pprint(analysis["excess_return_without_cost"])
        pprint(f"以下是有成本超额收益的分析结果({analysis_freq})。")
        pprint(analysis["excess_return_with_cost"])


结果
------

回测结果格式如下：

.. code-block:: python

                                                      risk
    excess_return_without_cost mean               0.000605
                               std                0.005481
                               annualized_return  0.152373
                               information_ratio  1.751319
                               max_drawdown      -0.059055
    excess_return_with_cost    mean               0.000410
                               std                0.005478
                               annualized_return  0.103265
                               information_ratio  1.187411
                               max_drawdown      -0.075024


- `excess_return_without_cost`（无成本超额收益）
    - `mean`（均值）
        无成本`CAR`（累计超额收益）的平均值
    - `std`（标准差）
        无成本`CAR`（累计超额收益）的标准差
    - `annualized_return`（年化收益率）
        无成本`CAR`（累计超额收益）的年化收益率
    - `information_ratio`（信息比率）
        无成本信息比率，详情请参考`Information Ratio – IR <https://www.investopedia.com/terms/i/informationratio.asp>`_。
    - `max_drawdown`（最大回撤）
        无成本`CAR`（累计超额收益）的最大回撤，详情请参考`Maximum Drawdown (MDD) <https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp>`_。

- `excess_return_with_cost`（有成本超额收益）
    - `mean`（均值）
        有成本`CAR`（累计超额收益）序列的平均值
    - `std`（标准差）
        有成本`CAR`（累计超额收益）序列的标准差
    - `annualized_return`（年化收益率）
        有成本`CAR`（累计超额收益）的年化收益率
    - `information_ratio`（信息比率）
        有成本信息比率，详情请参考`Information Ratio – IR <https://www.investopedia.com/terms/i/informationratio.asp>`_。
    - `max_drawdown`（最大回撤）
        有成本`CAR`（累计超额收益）的最大回撤，详情请参考`Maximum Drawdown (MDD) <https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp>`_。


参考
=========
要了解更多关于``预测模型``输出的`prediction score`（预测分数）`pred_score`的信息，请参考`预测模型：模型训练与预测 <model.html>`_。
