# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Base exception class
class QlibException(Exception):
    pass


class RecorderInitializationError(QlibException):
    """实验开始时重新初始化的错误类型"""


class LoadObjectError(QlibException):
    """Recorder无法加载对象时的错误类型"""


class ExpAlreadyExistError(Exception):
    """实验已存在"""
