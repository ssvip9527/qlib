.. _api:

=============
API 参考
=============



在这里你可以找到所有 ``Qlib`` 的接口。


数据
====

数据提供器
----------

.. automodule:: qlib.data.data
    :members:

过滤器
------

.. automodule:: qlib.data.filter
    :members:

类
-----
.. automodule:: qlib.data.base
    :members:

操作符
--------
.. automodule:: qlib.data.ops
    :members:

缓存
-----
.. autoclass:: qlib.data.cache.MemCacheUnit
    :members:

.. autoclass:: qlib.data.cache.MemCache
    :members:

.. autoclass:: qlib.data.cache.ExpressionCache
    :members:

.. autoclass:: qlib.data.cache.DatasetCache
    :members:

.. autoclass:: qlib.data.cache.DiskExpressionCache
    :members:

.. autoclass:: qlib.data.cache.DiskDatasetCache
    :members:


存储
-------
.. autoclass:: qlib.data.storage.storage.BaseStorage
    :members:

.. autoclass:: qlib.data.storage.storage.CalendarStorage
    :members:

.. autoclass:: qlib.data.storage.storage.InstrumentStorage
    :members:

.. autoclass:: qlib.data.storage.storage.FeatureStorage
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileStorageMixin
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileCalendarStorage
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileInstrumentStorage
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileFeatureStorage
    :members:


数据集
-------

数据集类
~~~~~~~~~~~~~
.. automodule:: qlib.data.dataset.__init__
    :members:

数据加载器
~~~~~~~~~~~
.. automodule:: qlib.data.dataset.loader
    :members:

数据处理器
~~~~~~~~~~~~
.. automodule:: qlib.data.dataset.handler
    :members:

处理器
~~~~~~~~~
.. automodule:: qlib.data.dataset.processor
    :members:


扩展
=======

模型
-----
.. automodule:: qlib.model.base
    :members:

策略
--------

.. automodule:: qlib.contrib.strategy
    :members:

评估
--------

.. automodule:: qlib.contrib.evaluate
    :members:


报告
------

.. automodule:: qlib.contrib.report.analysis_position.report
    :members:



.. automodule:: qlib.contrib.report.analysis_position.score_ic
    :members:



.. automodule:: qlib.contrib.report.analysis_position.cumulative_return
    :members:



.. automodule:: qlib.contrib.report.analysis_position.risk_analysis
    :members:



.. automodule:: qlib.contrib.report.analysis_position.rank_label
    :members:



.. automodule:: qlib.contrib.report.analysis_model.analysis_model_performance
    :members:


工作流
========


实验管理器
------------------
.. autoclass:: qlib.workflow.expm.ExpManager
    :members:

实验
----------
.. autoclass:: qlib.workflow.exp.Experiment
    :members:

记录器
--------
.. autoclass:: qlib.workflow.recorder.Recorder
    :members:

记录模板
---------------
.. automodule:: qlib.workflow.record_temp
    :members:

任务管理
===============


任务生成器
----------
.. automodule:: qlib.workflow.task.gen
    :members:

任务管理器
-----------
.. automodule:: qlib.workflow.task.manage
    :members:

训练器
-------
.. automodule:: qlib.model.trainer
    :members:

收集器
---------
.. automodule:: qlib.workflow.task.collect
    :members:

分组
-----
.. automodule:: qlib.model.ens.group
    :members:

集成
--------
.. automodule:: qlib.model.ens.ensemble
    :members:

工具
-----
.. automodule:: qlib.workflow.task.utils
    :members:


在线服务
==============


在线管理器
--------------
.. automodule:: qlib.workflow.online.manager
    :members:

在线策略
---------------
.. automodule:: qlib.workflow.online.strategy
    :members:

在线工具
-----------
.. automodule:: qlib.workflow.online.utils
    :members:


记录更新器
-------------
.. automodule:: qlib.workflow.online.update
    :members:


工具
=====

可序列化
------------

.. automodule:: qlib.utils.serial
    :members:

强化学习
==============

基础组件
--------------
.. automodule:: qlib.rl
    :members:
    :imported-members:

策略
--------
.. automodule:: qlib.rl.strategy
    :members:
    :imported-members:

训练器
-------
.. automodule:: qlib.rl.trainer
    :members:
    :imported-members:

订单执行
---------------
.. automodule:: qlib.rl.order_execution
    :members:
    :imported-members:

工具
---------------
.. automodule:: qlib.rl.utils
    :members:
    :imported-members: