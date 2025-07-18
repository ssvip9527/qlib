
========
使用指南
========
.. currentmodule:: qlib

QlibRL可以帮助用户快速入门并便捷地实现基于强化学习（RL）算法的量化策略。针对不同用户群体，我们推荐以下使用QlibRL的指南。

强化学习算法初学者
==============================================
无论您是想了解RL在交易中的应用的量化研究者，还是想在交易场景中入门RL算法的学习者，如果您对RL了解有限并希望屏蔽各种详细设置以快速入门RL算法，我们推荐按以下顺序学习qlibrl：
 - 在`第一部分 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#reinforcement-learning>`_ 学习强化学习的基础知识。
 - 在`第二部分 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#potential-application-scenarios-in-quantitative-trading>`_ 了解RL方法可应用的交易场景。
 - 在`第三部分 <https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html>`_ 运行示例，使用RL解决交易问题。
 - 如果您想进一步探索QlibRL并进行一些定制，需要先在`第四部分 <https://qlib.readthedocs.io/en/latest/component/rl/framework.html>`_ 了解QlibRL的框架，然后根据需要重写特定组件。

强化学习算法研究者
==============================================
如果您已经熟悉现有的RL算法并致力于RL算法研究，但缺乏金融领域的专业知识，并且希望在金融交易场景中验证您的算法有效性，我们推荐以下步骤入门QlibRL：
 - 在`第二部分 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#potential-application-scenarios-in-quantitative-trading>`_ 了解RL方法可应用的交易场景。
 - 选择一个RL应用场景（目前QlibRL已实现两个场景示例：订单执行和算法交易）。在`第三部分 <https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html>`_ 运行示例使其工作。
 - 修改`policy <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/policy.py>`_ 部分以整合您自己的RL算法。

量化研究者
=======================
如果您具备一定的金融领域知识和编码技能，并希望探索RL算法在投资领域的应用，我们推荐以下步骤探索QlibRL：
 - 在`第一部分 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#reinforcement-learning>`_ 学习强化学习的基础知识。
 - 在`第二部分 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#potential-application-scenarios-in-quantitative-trading>`_ 了解RL方法可应用的交易场景。
 - 在`第三部分 <https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html>`_ 运行示例，使用RL解决交易问题。
 - Understand the framework of QlibRL in `part4 <https://qlib.readthedocs.io/en/latest/component/rl/framework.html>`_.
 - 根据您要解决的问题的特点选择合适的RL算法（目前QlibRL支持基于tianshou的PPO和DQN算法）。
 - 根据市场交易规则和您要解决的问题设计MDP（马尔可夫决策过程）流程。参考订单执行中的示例，对以下模块进行相应修改：`State <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/state.py#L70>`_、`Metrics <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/state.py#L18>`_、`ActionInterpreter <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/interpreter.py#L199>`_、`StateInterpreter <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/interpreter.py#L68>`_、`Reward <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/reward.py>`_、`Observation <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/interpreter.py#L44>`_、`Simulator <https://github.com/ssvip9527/qlib/blob/main/qlib/rl/order_execution/simulator_simple.py>`_。