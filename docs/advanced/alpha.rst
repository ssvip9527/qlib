.. _alpha:

=========================
构建公式化Alpha因子
=========================
.. currentmodule:: qlib

简介
============

在量化交易实践中，设计能够解释和预测未来资产收益的新型因子对策略的盈利能力至关重要。这类因子通常被称为alpha因子，简称alpha。


公式化alpha，顾名思义，是一种可以表示为公式或数学表达式的alpha因子。


在``Qlib``中构建公式化Alpha因子
=====================================

In ``Qlib``, users can easily build formulaic alphas.

示例
-------

`MACD`（移动平均收敛散度）是股票价格技术分析中使用的一种公式化alpha因子，旨在揭示股票价格趋势的强度、方向、动量和持续时间的变化。

`MACD`可以表示为以下公式：

.. math::

    MACD = 2\times (DIF-DEA)

.. note::

    `DIF`表示差离值，即12期指数移动平均线(EMA)减去26期指数移动平均线。

    .. math::

        DIF = \frac{EMA(CLOSE, 12) - EMA(CLOSE, 26)}{CLOSE}

    `DEA`表示DIF的9期指数移动平均线。

    .. math::

        DEA = \frac{EMA(DIF, 9)}{CLOSE}

用户可以使用``Data Handler``在Qlib中构建公式化alpha因子`MACD`：

.. note:: 用户需要先使用`qlib.init`初始化``Qlib``。请参考`初始化 <../start/initialization.html>`_。

.. code-block:: python

    >> from qlib.data.dataset.loader import QlibDataLoader
    >> MACD_EXP = '(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'
    >> fields = [MACD_EXP] # MACD
    >> names = ['MACD']
    >> labels = ['Ref($close, -2)/Ref($close, -1) - 1'] # label
    >> label_names = ['LABEL']
    >> data_loader_config = {
    ..     "feature": (fields, names),
    ..     "label": (labels, label_names)
    .. }
    >> data_loader = QlibDataLoader(config=data_loader_config)
    >> df = data_loader.load(instruments='csi300', start_time='2010-01-01', end_time='2017-12-31')
    >> print(df)
                            feature     label
                               MACD     LABEL
    datetime   instrument
    2010-01-04 SH600000   -0.011547 -0.019672
               SH600004    0.002745 -0.014721
               SH600006    0.010133  0.002911
               SH600008   -0.001113  0.009818
               SH600009    0.025878 -0.017758
    ...                         ...       ...
    2017-12-29 SZ300124    0.007306 -0.005074
               SZ300136   -0.013492  0.056352
               SZ300144   -0.000966  0.011853
               SZ300251    0.004383  0.021739
               SZ300315   -0.030557  0.012455

参考
=========

要了解有关``Data Loader``的更多信息，请参考`数据加载器 <../component/data.html#data-loader>`_

要了解有关``Data API``的更多信息，请参考`数据API <../component/data.html>`_
