.. _task_management:

===============
任务管理
===============
.. currentmodule:: qlib


简介
============

`工作流 <../component/introduction.html>`_ 部分介绍了如何以松耦合方式运行研究工作流。但使用 ``qrun`` 时只能执行一个 ``任务``。
为了自动生成和执行不同的任务，``任务管理`` 提供了包括 `任务生成`_、`任务存储`_、`任务训练`_ 和 `任务收集`_ 的完整流程。
通过此模块，用户可以在不同时间段、使用不同损失函数甚至不同模型自动运行其 ``任务``。任务生成、模型训练以及数据合并和收集的流程如下图所示。

.. image:: ../_static/img/Task-Gen-Recorder-Collector.svg
    :align: center

整个流程可用于 `在线服务 <../component/online.html>`_。

完整流程的示例见 `此处 <https://github.com/microsoft/qlib/tree/main/examples/model_rolling/task_manager_rolling.py>`__。

任务生成
===============
一个 ``任务`` 由 `模型`、`数据集`、`记录器` 或用户添加的任何内容组成。
具体的任务模板可在
`任务部分 <../component/workflow.html#task-section>`_ 中查看。
尽管任务模板是固定的，用户仍可以自定义 ``TaskGen``（任务生成器），通过任务模板生成不同的 ``任务``。

以下是 ``TaskGen`` 的基类：

.. autoclass:: qlib.workflow.task.gen.TaskGen
    :members:
    :noindex:

``Qlib`` 提供了一个 `RollingGen <https://github.com/microsoft/qlib/tree/main/qlib/workflow/task/gen.py>`_ 类，用于生成数据集在不同日期段的 ``任务`` 列表。
此类允许用户在一个实验中验证不同时期数据对模型的影响。更多信息见 `此处 <../reference/api.html#TaskGen>`__。

任务存储
============
为了提高效率并支持集群操作，``任务管理器`` 会将所有任务存储在 `MongoDB <https://www.mongodb.com/>`_ 中。
``TaskManager``（任务管理器）可以自动获取未完成的任务，并通过错误处理管理一组任务的生命周期。
使用此模块时，用户 **必须** 完成 `MongoDB <https://www.mongodb.com/>`_ 的配置。

用户需要在 `初始化 <../start/initialization.html#Parameters>`_ 中提供 MongoDB URL 和数据库名称以使用 ``TaskManager``，或进行如下配置：

    .. code-block:: python

        from qlib.config import C
        C["mongo"] = {
            "task_url" : "mongodb://localhost:27017/", # 你的 MongoDB URL
            "task_db_name" : "rolling_db" # 数据库名称
        }

.. autoclass:: qlib.workflow.task.manage.TaskManager
    :members:
    :noindex:

``任务管理器`` 的更多信息见 `此处 <../reference/api.html#TaskManager>`__。

任务训练
=============
生成并存储这些 ``任务`` 后，就可以运行处于 *WAITING*（等待）状态的 ``任务`` 了。
``Qlib`` 提供了 ``run_task`` 方法来运行任务池中的 ``任务``，不过用户也可以自定义任务的执行方式。
获取 ``task_func``（任务函数）的简单方法是直接使用 ``qlib.model.trainer.task_train``。
它将运行由 ``任务`` 定义的整个工作流，包括 *模型*、*数据集*、*记录器*。

.. autofunction:: qlib.workflow.task.manage.run_task
    :noindex:

同时，``Qlib`` 提供了一个名为 ``Trainer``（训练器）的模块。 

.. autoclass:: qlib.model.trainer.Trainer
    :members:
    :noindex:

``Trainer`` 会训练一系列任务并返回一系列模型记录器。
``Qlib`` 提供两种训练器：TrainerR 是最简单的方式，而 TrainerRM 基于 TaskManager，可帮助自动管理任务生命周期。
如果不想使用 ``任务管理器`` 管理任务，使用 TrainerR 训练由 ``TaskGen`` 生成的任务列表即可。
不同 ``Trainer`` 的详细信息见 `此处 <../reference/api.html#Trainer>`_。

任务收集
===============
收集模型训练结果前，需要使用 ``qlib.init`` 指定 mlruns 的路径。

为了收集训练后的 ``任务`` 结果，``Qlib`` 提供了 `Collector（收集器） <../reference/api.html#Collector>`_、`Group（分组器） <../reference/api.html#Group>`_ 和 `Ensemble（集成器） <../reference/api.html#Ensemble>`_，以可读、可扩展且松耦合的方式收集结果。

`Collector <../reference/api.html#Collector>`_ 可以从任何地方收集对象并对其进行合并、分组、平均等处理。它包含两个步骤：``collect``（将任何内容收集到字典中）和 ``process_collect``（处理收集到的字典）。

`Group <../reference/api.html#Group>`_ 也有两个步骤：``group``（可基于 `group_func` 对一组对象进行分组并转换为字典）和 ``reduce``（可基于某些规则将字典转换为集成结果）。
例如：{(A,B,C1): 对象, (A,B,C2): 对象} ---``group``---> {(A,B): {C1: 对象, C2: 对象}} ---``reduce``---> {(A,B): 对象}

`Ensemble <../reference/api.html#Ensemble>`_ 可以合并集成中的对象。
例如：{C1: 对象, C2: 对象} ---``Ensemble``---> 对象。
你可以在 ``Collector`` 的 process_list 中设置所需的集成器。
常见的集成器包括 ``AverageEnsemble``（平均集成器）和 ``RollingEnsemble``（滚动集成器）。平均集成器用于集成同一时间段内不同模型的结果，滚动集成器用于集成同一时间段内不同模型的结果。

因此，层次结构为：``Collector`` 的第二步对应 ``Group``，而 ``Group`` 的第二步对应 ``Ensemble``。

更多信息，请参见 `Collector <../reference/api.html#Collector>`_、`Group <../reference/api.html#Group>`_ 和 `Ensemble <../reference/api.html#Ensemble>`_，或 `示例 <https://github.com/microsoft/qlib/tree/main/examples/model_rolling/task_manager_rolling.py>`_。
