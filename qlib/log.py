# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import logging
from typing import Optional, Text, Dict, Any
import re
from logging import config as logging_config
from time import time
from contextlib import contextmanager

from .config import C


class MetaLogger(type):
    def __new__(mcs, name, bases, attrs):  # pylint: disable=C0204
        wrapper_dict = logging.Logger.__dict__.copy()
        for key, val in wrapper_dict.items():
            if key not in attrs and key != "__reduce__":
                attrs[key] = val
        return type.__new__(mcs, name, bases, attrs)


class QlibLogger(metaclass=MetaLogger):
    """
    Qlib的自定义日志器。
    """

    def __init__(self, module_name):
        self.module_name = module_name
        # this feature name conflicts with the attribute with Logger
        # rename it to avoid some corner cases that result in comparing `str` and `int`
        self.__level = 0

    @property
    def logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(self.__level)
        return logger

    def setLevel(self, level):
        self.__level = level

    def __getattr__(self, name):
        # During unpickling, python will call __getattr__. Use this line to avoid maximum recursion error.
        if name in {"__setstate__"}:
            raise AttributeError
        return self.logger.__getattribute__(name)


class _QLibLoggerManager:
    def __init__(self):
        self._loggers = {}

    def setLevel(self, level):
        for logger in self._loggers.values():
            logger.setLevel(level)

    def __call__(self, module_name, level: Optional[int] = None) -> QlibLogger:
        """
        获取特定模块的日志器。

        :param module_name: str
            逻辑模块名称。
        :param level: int
            日志级别
        :return: Logger
            日志器对象。
        """
        if level is None:
            level = C.logging_level

        if not module_name.startswith("qlib."):
            # 当请求的``module_name``不以``qlib.``开头时，添加qlib.前缀。
            # 如果module_name已经是qlib.xxx，则不在此格式化，否则会变成qlib.qlib.xxx。
            module_name = "qlib.{}".format(module_name)

        # Get logger.
        module_logger = self._loggers.setdefault(module_name, QlibLogger(module_name))
        module_logger.setLevel(level)
        return module_logger


get_module_logger = _QLibLoggerManager()


class TimeInspector:
    timer_logger = get_module_logger("timer")

    time_marks = []

    @classmethod
    def set_time_mark(cls):
        """
        用当前时间设置一个时间标记，并将此标记压入栈中。
        :return: float
            当前时间的时间戳。
        """
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        """
        从栈中弹出最后一个时间标记。
        """
        return cls.time_marks.pop()

    @classmethod
    def get_cost_time(cls):
        """
        从栈中获取最后一个时间标记，计算与当前时间的差值。
        :return: float
            最后一个时间标记与当前时间的差值。
        """
        cost_time = time() - cls.time_marks.pop()
        return cost_time

    @classmethod
    def log_cost_time(cls, info="Done"):
        """
        从栈中获取最后一个时间标记，计算与当前时间的差值，并记录时间差和信息。
        :param info: str
            将记录到标准输出的信息。
        """
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: {0:.3f}s | {1}".format(cost_time, info))

    @classmethod
    @contextmanager
    def logt(cls, name="", show_start=False):
        """logt.
        记录内部代码的执行时间

        参数
        ----------
        name :
            名称
        show_start :
            是否显示开始信息
        """
        if show_start:
            cls.timer_logger.info(f"{name} 开始")
        cls.set_time_mark()
        try:
            yield None
        finally:
            pass
        cls.log_cost_time(info=f"{name} 完成")


def set_log_with_config(log_config: Dict[Text, Any]):
    """使用配置设置日志

    :param log_config:
        日志配置字典
    :return:
        无
    """
    logging_config.dictConfig(log_config)


class LogFilter(logging.Filter):
    def __init__(self, param=None):
        super().__init__()
        self.param = param

    @staticmethod
    def match_msg(filter_str, msg):
        match = False
        try:
            if re.match(filter_str, msg):
                match = True
        except Exception:
            pass
        return match

    def filter(self, record):
        allow = True
        if isinstance(self.param, str):
            allow = not self.match_msg(self.param, record.msg)
        elif isinstance(self.param, list):
            allow = not any(self.match_msg(p, record.msg) for p in self.param)
        return allow


def set_global_logger_level(level: int, return_orig_handler_level: bool = False):
    """set qlib.xxx logger handlers level

    Parameters
    ----------
    level: int
        logger level

    return_orig_handler_level: bool
        return origin handler level map

    Examples
    ---------

        .. code-block:: python

            import qlib
            import logging
            from qlib.log import get_module_logger, set_global_logger_level
            qlib.init()

            tmp_logger_01 = get_module_logger("tmp_logger_01", level=logging.INFO)
            tmp_logger_01.info("1. tmp_logger_01 info show")

            global_level = logging.WARNING + 1
            set_global_logger_level(global_level)
            tmp_logger_02 = get_module_logger("tmp_logger_02", level=logging.INFO)
            tmp_logger_02.log(msg="2. tmp_logger_02 log show", level=global_level)

            tmp_logger_01.info("3. tmp_logger_01 info do not show")

    """
    _handler_level_map = {}
    qlib_logger = logging.root.manager.loggerDict.get("qlib", None)  # pylint: disable=E1101
    if qlib_logger is not None:
        for _handler in qlib_logger.handlers:
            _handler_level_map[_handler] = _handler.level
            _handler.level = level
    return _handler_level_map if return_orig_handler_level else None


@contextmanager
def set_global_logger_level_cm(level: int):
    """set qlib.xxx logger handlers level to use contextmanager

    Parameters
    ----------
    level: int
        logger level

    Examples
    ---------

        .. code-block:: python

            import qlib
            import logging
            from qlib.log import get_module_logger, set_global_logger_level_cm
            qlib.init()

            tmp_logger_01 = get_module_logger("tmp_logger_01", level=logging.INFO)
            tmp_logger_01.info("1. tmp_logger_01 info show")

            global_level = logging.WARNING + 1
            with set_global_logger_level_cm(global_level):
                tmp_logger_02 = get_module_logger("tmp_logger_02", level=logging.INFO)
                tmp_logger_02.log(msg="2. tmp_logger_02 log show", level=global_level)
                tmp_logger_01.info("3. tmp_logger_01 info do not show")

            tmp_logger_01.info("4. tmp_logger_01 info show")

    """
    _handler_level_map = set_global_logger_level(level, return_orig_handler_level=True)
    try:
        yield
    finally:
        for _handler, _level in _handler_level_map.items():
            _handler.level = _level
