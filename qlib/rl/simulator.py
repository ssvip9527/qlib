# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from .seed import InitialStateType

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper

StateType = TypeVar("StateType")
"""StateType存储模拟过程中的所有有用数据
(以及在需要时生成/检索数据的实用工具)。"""

ActType = TypeVar("ActType")
"""此ActType是模拟器端动作的类型。"""


class Simulator(Generic[InitialStateType, StateType, ActType]):
    """
    通过``__init__``重置并通过``step(action)``转换的模拟器。

    为了使数据流清晰，我们对模拟器做了以下限制：

    1. 修改模拟器内部状态的唯一方法是使用``step(action)``。
    2. 外部模块可以通过``simulator.get_state()``*读取*模拟器的状态，
       并通过调用``simulator.done()``检查模拟器是否处于结束状态。

    模拟器被定义为与三种类型绑定：

    - *InitialStateType*：用于创建模拟器的数据类型
    - *StateType*：模拟器**状态**的类型
    - *ActType*：**动作**的类型，即每一步接收的输入

    不同的模拟器可能共享相同的StateType。例如，当它们处理相同的任务但使用不同的模拟实现时。
    通过相同的类型，它们可以安全地共享MDP中的其他组件。

    模拟器是短暂的。模拟器的生命周期从初始状态开始，到轨迹结束为止。
    换句话说，当轨迹结束时，模拟器会被回收。
    如果模拟器之间需要共享上下文(例如为了加速)，
    可以通过访问环境包装器的弱引用来实现。

    属性
    ----------
    env
        环境包装器的引用，在某些特殊情况下可能有用。
        不建议模拟器使用此属性，因为它容易引发错误。
    """

    env: Optional[EnvWrapper] = None

    def __init__(self, initial: InitialStateType, **kwargs: Any) -> None:
        pass

    def step(self, action: ActType) -> None:
        """接收一个ActType类型的动作。

        模拟器应更新其内部状态，并返回None。
        更新后的状态可以通过``simulator.get_state()``获取。
        """
        raise NotImplementedError()

    def get_state(self) -> StateType:
        raise NotImplementedError()

    def done(self) -> bool:
        """检查模拟器是否处于"done"状态。当模拟器处于"done"状态时，
        它不应再接收任何``step``请求。由于模拟器是短暂的，要重置模拟器，应销毁旧的模拟器并创建一个新的。
        """
        raise NotImplementedError()
