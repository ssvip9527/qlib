.. _online_serving:

==============
在线服务
==============
.. currentmodule:: qlib


介绍
============

.. image:: ../_static/img/online_serving.png
    :align: center

除了回测之外，测试模型有效性的一种方法是在真实市场条件下进行预测，甚至基于这些预测进行真实交易。
``Online Serving``是一组使用最新数据的在线模型模块，
包括``在线管理器 <#Online Manager>``_、`在线策略 <#Online Strategy>`_、`在线工具 <#Online Tool>`_和`更新器 <#Updater>`_。

`这里 <https://github.com/ssvip9527/qlib/tree/main/examples/online_srv>`_ 有几个参考示例，展示了``Online Serving``的不同功能。
如果您有许多模型或``任务``需要管理，请考虑使用``任务管理 <../advanced/task_management.html>``_。
这些``示例 <https://github.com/ssvip9527/qlib/tree/main/examples/online_srv>``_基于``任务管理 <../advanced/task_management.html>``_中的一些组件，如``TrainerRM``或``Collector``。

**注意**：用户应保持数据源更新以支持在线服务。例如，Qlib提供了``一组脚本 <https://github.com/ssvip9527/qlib/blob/main-cn/scripts/data_collector/yahoo/README.md#automatic-update-of-daily-frequency-datafrom-yahoo-finance>``_来帮助用户更新Yahoo的每日数据。

当前已知限制
- 目前支持对下一个交易日的每日更新预测。但由于`公开数据的限制 <https://github.com/ssvip9527/qlib/issues/215#issuecomment-766293563>_`，不支持为下一个交易日生成订单。


在线管理器
==============

.. automodule:: qlib.workflow.online.manager
    :members:
    :noindex:

在线策略
===============

.. automodule:: qlib.workflow.online.strategy
    :members:
    :noindex:

在线工具
===========

.. automodule:: qlib.workflow.online.utils
    :members:
    :noindex:

更新器
=======

.. automodule:: qlib.workflow.online.update
    :members:
    :noindex:
