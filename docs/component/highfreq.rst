.. _highfreq:

========================================================================
高频交易嵌套决策执行框架设计
========================================================================
.. currentmodule:: qlib

介绍
============

日频交易(如投资组合管理)和日内交易(如订单执行)是量化投资中的两个热门话题，通常被分开研究。

要获得日频和日内交易的联合表现，它们必须相互交互并联合进行回测。
为了支持多层次的联合回测策略，需要一个相应的框架。目前公开的高频交易框架都没有考虑多层次联合交易，这使得上述回测不准确。

除了回测，不同层次策略的优化也不是独立的，会相互影响。
例如，最佳的投资组合管理策略可能会随着订单执行性能的变化而变化(例如，当我们改进订单执行策略时，周转率更高的投资组合可能会成为更好的选择)。
为了获得整体良好的性能，有必要考虑不同层次策略之间的相互作用。

因此，为了解决上述各种问题，构建一个新的多层次交易框架变得必要，为此我们设计了一个考虑策略交互的嵌套决策执行框架。

.. image:: ../_static/img/framework.svg

该框架的设计如上图中间的黄色部分所示。每个层次由``Trading Agent``和``Execution Env``组成。``Trading Agent``有自己的数据处理模块(``Information Extractor``)、预测模块(``Forecast Model``)和决策生成器(``Decision Generator``)。交易算法通过``Decision Generator``基于``Forecast Module``输出的预测信号生成决策，交易算法生成的决策传递给``Execution Env``，后者返回执行结果。

交易算法的频率、决策内容和执行环境可以由用户定制(如日内交易、日频交易、周频交易)，执行环境内部可以嵌套更细粒度的交易算法和执行环境(即图中的子工作流，例如日频订单可以通过在日内拆分订单变成更细粒度的决策)。嵌套决策执行框架的灵活性使用户能够轻松探索不同层次交易策略组合的效果，并打破交易算法不同层次之间的优化壁垒。

嵌套决策执行框架的优化可以在`QlibRL <./rl/overall.html>`_的支持下实现。要了解更多关于如何使用QlibRL的信息，请参阅API参考：`RL API <../reference/api.html#rl>`_。

示例
=======

高频嵌套决策执行框架的示例可以在`这里 <https://github.com/ssvip9527/qlib/blob/main/examples/nested_decision_execution/workflow.py>`_找到。


此外，除了上述示例，这里还有一些Qlib中关于高频交易的其他相关工作。

- `Prediction with high-frequency data <https://github.com/ssvip9527/qlib/tree/main/examples/highfreq#benchmarks-performance-predicting-the-price-trend-in-high-frequency-data>`_
- `Examples <https://github.com/ssvip9527/qlib/blob/main/examples/orderbook_data/>`_ to extract features from high-frequency data without fixed frequency.
- `A paper <https://github.com/ssvip9527/qlib/tree/high-freq-execution#high-frequency-execution>`_ for high-frequency trading.
