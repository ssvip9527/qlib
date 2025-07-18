# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import atexit
import logging
import sys
import traceback

from ..log import get_module_logger
from . import R
from .recorder import Recorder

logger = get_module_logger("workflow", logging.INFO)


# function to handle the experiment when unusual program ending occurs
def experiment_exit_handler():
    """
    处理程序异常结束时实验的方法。
    `atexit`处理程序应放在最后，因为只要程序结束就会被调用。
    因此，如果之前发生任何异常或用户中断，我们应该先处理它们。一旦`R`结束，
    再次调用`R.end_exp`将不会生效。

    限制：
    - 如果在程序中使用pdb，结束时不会触发excepthook。状态将被标记为已完成
    """
    sys.excepthook = experiment_exception_hook  # handle uncaught exception
    atexit.register(R.end_exp, recorder_status=Recorder.STATUS_FI)  # will not take effect if experiment ends


def experiment_exception_hook(exc_type, value, tb):
    """
    以"FAILED"状态结束实验。此异常处理尝试捕获未捕获的异常
    并自动结束实验。

    参数
    exc_type: 异常类型
    value: 异常值
    tb: 异常回溯
    """
    logger.error(f"An exception has been raised[{exc_type.__name__}: {value}].")

    # Same as original format
    traceback.print_tb(tb)
    print(f"{exc_type.__name__}: {value}")

    R.end_exp(recorder_status=Recorder.STATUS_FA)
