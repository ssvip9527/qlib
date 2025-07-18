===============================
``Qlib``: 量化投资平台
===============================

简介
============

.. image:: ../_static/img/logo/white_bg_rec+word.png
    :align: center

``Qlib``是一个面向人工智能的量化投资平台，旨在发掘AI技术在量化投资中的潜力，赋能量化研究，创造AI技术的量化投资价值。

借助``Qlib``，用户可以轻松尝试各种想法，创建更优的量化投资策略。

框架
=========


.. image:: ../_static/img/framework.svg
    :align: center


在模块层面，Qlib是一个由上述组件构成的平台。这些组件被设计为松耦合模块，每个组件都可以独立使用。

对于Qlib的新用户来说，这个框架可能有些复杂。它试图准确包含Qlib设计的许多细节。
新用户可以先跳过本节，稍后再阅读。



===========================  ==============================================================================
名称                         描述
===========================  ==============================================================================
`基础设施`层                 `基础设施`层为量化研究提供底层支持。
                             `DataServer`为用户提供高性能的基础设施，用于管理和检索原始数据。`Trainer`提供灵活的接口来控制模型的训练过程，使算法能够控制训练流程。

`学习框架`层                 `预测模型`和`交易智能体`是可训练的。它们基于`学习框架`层进行训练，然后应用于`工作流`层的多个场景。支持的学习范式可分为强化学习和监督学习。学习框架也利用`工作流`层（例如共享`信息提取器`，基于`执行环境`创建环境）。

`工作流`层                   `工作流`层涵盖量化投资的整个工作流程。支持基于监督学习的策略和基于强化学习的策略。
                             `信息提取器`为模型提取数据。`预测模型`专注于为其他模块生成各种预测信号（例如*alpha*、风险）。利用这些信号，`决策生成器`将生成目标交易决策（即投资组合、订单）。
                             如果采用基于强化学习的策略，`策略`将以端到端的方式学习，直接生成交易决策。
                             决策将由`执行环境`（即交易市场）执行。可能存在多个级别的`策略`和`执行器`（例如，*订单执行交易策略和日内订单执行器*可以表现为日间交易循环，并嵌套在*日度组合管理交易策略和日间交易执行器*交易循环中）

`接口`层                     `接口`层试图为底层系统提供用户友好的界面。`分析器`模块将为用户提供关于预测信号、投资组合和执行结果的详细分析报告
===========================  ==============================================================================

- 手绘风格的模块正在开发中，将在未来发布。
- 虚线边框的模块具有高度的用户可定制性和可扩展性。

（注：框架图使用https://draw.io/创建）
