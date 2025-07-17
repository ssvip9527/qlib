# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""定义一组初始状态定义和状态集定义。

在仅执行单资产订单的情况下，唯一的种子是订单。
"""

from typing import TypeVar

InitialStateType = TypeVar("InitialStateType")
"""用于创建模拟器的数据类型。"""
