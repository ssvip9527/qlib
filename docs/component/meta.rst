.. _meta:

======================================================
元控制器：元任务 & 元数据集 & 元模型
======================================================
.. currentmodule:: qlib


介绍
============
``Meta Controller``为``Forecast Model``提供指导，旨在学习一系列预测任务中的规律模式，并使用学习到的模式来指导即将到来的预测任务。用户可以基于``Meta Controller``模块实现自己的元模型实例。

元任务
=========

`Meta Task`实例是元学习框架中的基本元素。它保存了可用于`Meta Model`的数据。多个`Meta Task`实例可能共享同一个`Data Handler`，由`Meta Dataset`控制。用户应使用`prepare_task_data()`获取可以直接输入`Meta Model`的数据。

.. autoclass:: qlib.model.meta.task.MetaTask
    :members:

元数据集
============

`Meta Dataset`控制元信息生成过程。它负责为训练`Meta Model`提供数据。用户应使用`prepare_tasks`获取`Meta Task`实例列表。

.. autoclass:: qlib.model.meta.dataset.MetaTaskDataset
    :members:

元模型
==========

通用元模型
------------------
`Meta Model`实例是控制工作流程的部分。`Meta Model`的用法包括：
1. 用户使用`fit`函数训练他们的`Meta Model`。
2. `Meta Model`实例通过`inference`函数提供有用信息来指导工作流程。

.. autoclass:: qlib.model.meta.model.MetaModel
    :members:

元任务模型
---------------
这类元模型可以直接与任务定义交互。`Meta Task Model`是它们继承的类。它们通过修改基础任务定义来指导基础任务。函数`prepare_tasks`可用于获取修改后的基础任务定义。

.. autoclass:: qlib.model.meta.model.MetaTaskModel
    :members:

元指导模型
----------------
这类元模型参与基础预测模型的训练过程。元模型可以在基础预测模型训练期间指导它们以提高性能。

.. autoclass:: qlib.model.meta.model.MetaGuideModel
    :members:


示例
=======
``Qlib``提供了``Meta Model``模块的实现``DDG-DA``，
它适应市场动态。

``DDG-DA``包括四个步骤：

1. 计算元信息并将其封装到``Meta Task``实例中。所有元任务形成一个``Meta Dataset``实例。
2. 基于元数据集的训练数据训练``DDG-DA``。
3. 对``DDG-DA``进行推理以获取指导信息。
4. 将指导信息应用于预测模型以提高其性能。

上述示例可以在``examples/benchmarks_dynamic/DDG-DA/workflow.py``中找到。`这里 <https://github.com/microsoft/qlib/tree/main/examples/benchmarks_dynamic/DDG-DA>`_。
