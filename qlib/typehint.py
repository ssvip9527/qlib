# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Commonly used types."""

import sys
from typing import Union
from pathlib import Path

__all__ = ["Literal", "TypedDict", "final"]

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict, final  # type: ignore  # pylint: disable=no-name-in-module
else:
    from typing_extensions import Literal, TypedDict, final


class InstDictConf(TypedDict):
    """
    InstDictConf 是一个基于字典的配置，用于描述一个实例

        案例 1)
        {
            'class': 'ClassName',
            'kwargs': dict, # 可选参数。如果未提供，将使用{}
            'model_path': path, # 如果类中已指定模块，则可选
        }
        案例 2)
        {
            'class': <类本身>,
            'kwargs': dict, # 可选参数。如果未提供，将使用{}
        }
    """

    # class: str  # because class is a keyword of Python. We have to comment it
    kwargs: dict  # 可选参数。如果未提供，将使用{}
    module_path: str  # 如果类中已指定模块，则可选


InstConf = Union[InstDictConf, str, object, Path]
"""
InstConf是用于描述实例的类型；它将被传入Qlib的init_instance_by_config函数

    config : Union[str, dict, object, Path]

        InstDictConf示例：
            请参考InstDictConf的文档

        字符串示例：
            1) 指定pickle对象
                - 路径格式如'file:///<pickle文件路径>/obj.pkl'
            2) 指定类名
                - "ClassName": 将使用getattr(module, "ClassName")()
            3) 指定包含类名的模块路径
                - "a.b.c.ClassName": 将使用getattr(<a.b.c.module>, "ClassName")()

        对象示例：
            accept_types的实例

        Path示例：
            指定pickle对象
                - 将被视为'file:///<pickle文件路径>/obj.pkl'
"""
