QlibRL框架
=======================

QlibRL包含一整套组件，涵盖了强化学习pipeline的整个生命周期，包括构建市场模拟器、状态与动作塑造、策略（战略）训练以及在模拟环境中回测策略。

QlibRL主要基于Tianshou和Gym框架实现。QlibRL的高层结构如下所示：

.. image:: ../../_static/img/QlibRL_framework.png
   :width: 600
   :align: center

这里，我们简要介绍图中的每个组件。

EnvWrapper
------------
EnvWrapper是模拟环境的完整封装。它接收外部（策略/战略/智能体）的动作，模拟市场变化，然后返回奖励和更新后的状态，从而形成一个交互循环。

在QlibRL中，EnvWrapper是gym.Env的子类，因此它实现了gym.Env的所有必要接口。任何接受gym.Env的类或流水线也应该接受EnvWrapper。开发人员无需实现自己的EnvWrapper来构建环境，只需实现EnvWrapper的4个组件：

- `Simulator`
    模拟器是负责环境模拟的核心组件。开发人员可以以任何方式在Simulator中实现与环境模拟直接相关的所有逻辑。在QlibRL中，已经有两个针对单一资产交易的Simulator实现：1) ``SingleAssetOrderExecution``，基于Qlib的回测工具包构建，考虑了许多实际交易细节但速度较慢。2) ``SimpleSingleAssetOrderExecution``，基于简化的交易模拟器构建，忽略了许多细节（如交易限制、四舍五入）但速度很快。
- `State interpreter` 
    状态解释器负责将原始格式（模拟器提供的格式）的状态“解释”为策略可以理解的格式。例如，将非结构化原始特征转换为数值张量。
- `Action interpreter` 
    动作解释器类似于状态解释器。但它不是解释状态，而是将策略生成的动作从策略提供的格式解释为模拟器可接受的格式。
- `Reward function` 
    奖励函数在策略每次执行动作后向策略返回数值奖励。 

EnvWrapper会有机地组织这些组件。这种分解使开发具有更好的灵活性。例如，如果开发人员想在同一环境中训练多种类型的策略，只需设计一个模拟器，并为不同类型的策略设计不同的状态解释器/动作解释器/奖励函数。

QlibRL为这4个组件都定义了完善的基类。开发人员只需通过继承基类并实现基类所需的所有接口来定义自己的组件。上述基础组件的API可在`这里 <../../reference/api.html#module-qlib.rl>`__找到。

Policy
------------
QlibRL直接使用Tianshou的策略。开发人员可以直接使用Tianshou提供的策略，或通过继承Tianshou的策略实现自己的策略。

Training Vessel & Trainer
-------------------------
顾名思义，训练容器（training vessel）和训练器（trainer）是训练中使用的辅助类。训练容器包含模拟器/解释器/奖励函数/策略，并控制训练的算法相关部分。相应地，训练器负责控制训练的运行时部分。

您可能已经注意到，训练容器本身持有构建EnvWrapper所需的所有组件，而不是直接持有EnvWrapper实例。这允许训练容器在必要时动态创建EnvWrapper的副本（例如，在并行训练情况下）。

通过训练容器，训练器最终可以通过简单的、类Scikit-learn的接口（即``trainer.fit()``）启动训练流水线。

Trainer和TrainingVessel的API可在`这里 <../../reference/api.html#module-qlib.rl.trainer>`__找到。

RL模块采用松耦合设计。目前，RL示例与具体业务逻辑集成，但RL的核心部分比您看到的要简单得多。为了展示RL的简单核心，我们创建了`一个专用笔记本 <https://github.com/ssvip9527/qlib/tree/main/examples/rl/simple_example.ipynb>`__，用于无业务损失的RL演示。
