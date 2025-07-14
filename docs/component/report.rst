.. _report:

=======================================
分析：评估与结果分析
=======================================

简介
============

``Analysis`` 模块用于展示日内交易的图形化报告，帮助用户直观地评估和分析投资组合。以下是可查看的图表类型：

- analysis_position
    - report_graph（报告图表）
    - score_ic_graph（IC值图表）
    - cumulative_return_graph（累计收益图表）
    - risk_analysis_graph（风险分析图表）
    - rank_label_graph（排名标签图表）

- analysis_model
    - model_performance_graph（模型性能图表）


Qlib中所有累积收益指标（如收益率、最大回撤）均通过求和方式计算。
这避免了指标或图表随时间呈指数级扭曲。

图形化报告
=================

用户可以运行以下代码获取所有支持的报告：

.. code-block:: python

    >> import qlib.contrib.report as qcr
    >> print(qcr.GRAPH_NAME_LIST)
    ['analysis_position.report_graph', 'analysis_position.score_ic_graph', 'analysis_position.cumulative_return_graph', 'analysis_position.risk_analysis_graph', 'analysis_position.rank_label_graph', 'analysis_model.model_performance_graph']

.. note::

    有关更多详细信息，请参考函数文档：类似 ``help(qcr.analysis_position.report_graph)`` 的用法



Usage & Example
===============

Usage of `analysis_position.report`
-----------------------------------

API
~~~

.. automodule:: qlib.contrib.report.analysis_position.report
    :members:
    :noindex:

Graphical Result
~~~~~~~~~~~~~~~~

.. note::

    - 横轴 X：交易日
    - 纵轴 Y：
        - `cum bench`
            基准的累计收益序列
        - `cum return wo cost`
            无成本的投资组合累计收益序列
        - `cum return w cost`
            有成本的投资组合累计收益序列
        - `return wo mdd`
            无成本累计收益的最大回撤序列
        - `return w cost mdd`:
            有成本累计收益的最大回撤序列
        - `cum ex return wo cost`
            无成本的投资组合相对基准的累计超额收益（CAR）序列
        - `cum ex return w cost`
            有成本的投资组合相对基准的累计超额收益（CAR）序列
        - `turnover`
            换手率序列
        - `cum ex return wo cost mdd`
            无成本累计超额收益（CAR）的回撤序列
        - `cum ex return w cost mdd`
            有成本累计超额收益（CAR）的回撤序列
    - 上方阴影部分：对应 `cum return wo cost` 的最大回撤
    - 下方阴影部分：对应 `cum ex return wo cost` 的最大回撤

.. image:: ../_static/img/analysis/report.png


Usage of `analysis_position.score_ic`
-------------------------------------

API
~~~

.. automodule:: qlib.contrib.report.analysis_position.score_ic
    :members:
    :noindex:


Graphical Result
~~~~~~~~~~~~~~~~

.. note::

    - 横轴 X：交易日
    - 纵轴 Y：
        - `ic`
            `label` 与 `prediction score` 之间的皮尔逊相关系数序列。
            在上述示例中，`label` 的计算公式为 `Ref($close, -2)/Ref($close, -1)-1`。更多详情请参考 `数据特征 <data.html#feature>`_。

        - `rank_ic`
            `label` 与 `prediction score` 之间的斯皮尔曼等级相关系数序列。

.. image:: ../_static/img/analysis/score_ic.png


.. Usage of `analysis_position.cumulative_return`
.. ----------------------------------------------
..
.. API
.. ~~~~~~~~~~~~~~~~
..
.. .. automodule:: qlib.contrib.report.analysis_position.cumulative_return
..     :members:
..
.. Graphical Result
.. ~~~~~~~~~~~~~~~~~
..
.. .. note::
..
..     - Axis X: Trading day
..     - Axis Y:
..         - Above axis Y: `(((Ref($close, -1)/$close - 1) * weight).sum() / weight.sum()).cumsum()`
..         - Below axis Y: Daily weight sum
..     - In the **sell** graph, `y < 0` stands for profit; in other cases, `y > 0` stands for profit.
..     - In the **buy_minus_sell** graph, the **y** value of the **weight** graph at the bottom is `buy_weight + sell_weight`.
..     - In each graph, the **red line** in the histogram on the right represents the average.
..
.. .. image:: ../_static/img/analysis/cumulative_return_buy.png
..
.. .. image:: ../_static/img/analysis/cumulative_return_sell.png
..
.. .. image:: ../_static/img/analysis/cumulative_return_buy_minus_sell.png
..
.. .. image:: ../_static/img/analysis/cumulative_return_hold.png


Usage of `analysis_position.risk_analysis`
------------------------------------------

API
~~~

.. automodule:: qlib.contrib.report.analysis_position.risk_analysis
    :members:
    :noindex:


Graphical Result
~~~~~~~~~~~~~~~~

.. note::

    - general graphics
        - `std`（标准差）
            - `excess_return_without_cost`
                无成本的累计超额收益（CAR）的标准差。
            - `excess_return_with_cost`
                有成本的累计超额收益（CAR）的标准差。
        - `annualized_return`（年化收益率）
            - `excess_return_without_cost`
                无成本的累计超额收益（CAR）的年化收益率。
            - `excess_return_with_cost`
                有成本的累计超额收益（CAR）的年化收益率。
        -  `information_ratio`（信息比率）
            - `excess_return_without_cost`
                无成本的信息比率。
            - `excess_return_with_cost`
                有成本的信息比率。

            有关信息比率的更多信息，请参考 `信息比率 - IR <https://www.investopedia.com/terms/i/informationratio.asp>`_。
        -  `max_drawdown`（最大回撤）
            - `excess_return_without_cost`
                无成本的累计超额收益（CAR）的最大回撤。
            - `excess_return_with_cost`
                有成本的累计超额收益（CAR）的最大回撤。


.. image:: ../_static/img/analysis/risk_analysis_bar.png
    :align: center

.. note::

    - annualized_return/max_drawdown/information_ratio/std 图表
        - 横轴 X：按月份分组的交易日
        - 纵轴 Y：
            - annualized_return 图表
                - `excess_return_without_cost_annualized_return`
                    无成本的月度累计超额收益（CAR）的年化收益率序列。
                - `excess_return_with_cost_annualized_return`
                    有成本的月度累计超额收益（CAR）的年化收益率序列。
            - max_drawdown 图表
                - `excess_return_without_cost_max_drawdown`
                    无成本的月度累计超额收益（CAR）的最大回撤序列。
                - `excess_return_with_cost_max_drawdown`
                    有成本的月度累计超额收益（CAR）的最大回撤序列。
            - information_ratio 图表
                - `excess_return_without_cost_information_ratio`
                    无成本的月度累计超额收益（CAR）的信息比率序列。
                - `excess_return_with_cost_information_ratio`
                    有成本的月度累计超额收益（CAR）的信息比率序列。
            - std 图表
                - `excess_return_without_cost_max_drawdown`
                    无成本的月度累计超额收益（CAR）的标准差序列。
                - `excess_return_with_cost_max_drawdown`
                    有成本的月度累计超额收益（CAR）的标准差序列。


.. image:: ../_static/img/analysis/risk_analysis_annualized_return.png
    :align: center

.. image:: ../_static/img/analysis/risk_analysis_max_drawdown.png
    :align: center

.. image:: ../_static/img/analysis/risk_analysis_information_ratio.png
    :align: center

.. image:: ../_static/img/analysis/risk_analysis_std.png
    :align: center

..
.. Usage of `analysis_position.rank_label`
.. ---------------------------------------
..
.. API
.. ~~~
..
.. .. automodule:: qlib.contrib.report.analysis_position.rank_label
..     :members:
..
..
.. Graphical Result
.. ~~~~~~~~~~~~~~~~
..
.. .. note::
..
    - hold/sell/buy 图表:
        - 横轴 X：交易日
        - 纵轴 Y：
            交易日内持有/卖出/买入股票的`label`的平均`排名比率`。

            在上述示例中，`label`的计算公式为`Ref($close, -1)/$close - 1`。`排名比率`的计算公式如下。
            .. math::

                ranking\ ratio = \frac{Ascending\ Ranking\ of\ label}{Number\ of\ Stocks\ in\ the\ Portfolio}
..
.. .. image:: ../_static/img/analysis/rank_label_hold.png
..     :align: center
..
.. .. image:: ../_static/img/analysis/rank_label_buy.png
..     :align: center
..
.. .. image:: ../_static/img/analysis/rank_label_sell.png
..     :align: center
..
..

Usage of `analysis_model.analysis_model_performance`
----------------------------------------------------

API
~~~

.. automodule:: qlib.contrib.report.analysis_model.analysis_model_performance
    :members:
    :noindex:


Graphical Results
~~~~~~~~~~~~~~~~~

.. note::

    - cumulative return graphics
        - `Group1`:
            The `Cumulative Return` series of stocks group with (`ranking ratio` of label <= 20%)
        - `Group2`:
            The `Cumulative Return` series of stocks group with (20% < `ranking ratio` of label <= 40%)
        - `Group3`:
            The `Cumulative Return` series of stocks group with (40% < `ranking ratio` of label <= 60%)
        - `Group4`:
            The `Cumulative Return` series of stocks group with (60% < `ranking ratio` of label <= 80%)
        - `Group5`:
            The `Cumulative Return` series of stocks group with (80% < `ranking ratio` of label)
        - `long-short`:
            The Difference series between `Cumulative Return` of `Group1` and of `Group5`
        - `long-average`
            The Difference series between `Cumulative Return` of `Group1` and average `Cumulative Return` for all stocks.

        The `ranking ratio` can be formulated as follows.
            .. math::

                ranking\ ratio = \frac{Ascending\ Ranking\ of\ label}{Number\ of\ Stocks\ in\ the\ Portfolio}

.. image:: ../_static/img/analysis/analysis_model_cumulative_return.png
    :align: center

.. note::
    - long-short/long-average
        The distribution of long-short/long-average returns on each trading day


.. image:: ../_static/img/analysis/analysis_model_long_short.png
    :align: center

.. TODO: ask xiao yang for detial

.. note::
    - Information Coefficient
        - The `Pearson correlation coefficient` series between `labels` and `prediction scores` of stocks in portfolio.
        - The graphics reports can be used to evaluate the `prediction scores`.

.. image:: ../_static/img/analysis/analysis_model_IC.png
    :align: center

.. note::
    - Monthly IC
        Monthly average of the `Information Coefficient`

.. image:: ../_static/img/analysis/analysis_model_monthly_IC.png
    :align: center

.. note::
    - IC
        The distribution of the `Information Coefficient` on each trading day.
    - IC Normal Dist. Q-Q
        The `Quantile-Quantile Plot` is used for the normal distribution of `Information Coefficient` on each trading day.

.. image:: ../_static/img/analysis/analysis_model_NDQ.png
    :align: center

.. note::
    - Auto Correlation
        - The `Pearson correlation coefficient` series between the latest `prediction scores` and the `prediction scores` `lag` days ago of stocks in portfolio on each trading day.
        - The graphics reports can be used to estimate the turnover rate.


.. image:: ../_static/img/analysis/analysis_model_auto_correlation.png
    :align: center
