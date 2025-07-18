# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import threading
from functools import partial
from threading import Thread
from typing import Callable, Text, Union

import joblib
from joblib import Parallel, delayed
from joblib._parallel_backends import MultiprocessingBackend
import pandas as pd

from queue import Empty, Queue
import concurrent

from qlib.config import C, QlibConfig


class ParallelExt(Parallel):
    def __init__(self, *args, **kwargs):
        maxtasksperchild = kwargs.pop("maxtasksperchild", None)
        super(ParallelExt, self).__init__(*args, **kwargs)
        if isinstance(self._backend, MultiprocessingBackend):
            # 2025-05-04 joblib released version 1.5.0, in which _backend_args was removed and replaced by _backend_kwargs.
            # Ref: https://github.com/joblib/joblib/pull/1525/files#diff-e4dff8042ce45b443faf49605b75a58df35b8c195978d4a57f4afa695b406bdc
            if joblib.__version__ < "1.5.0":
                self._backend_args["maxtasksperchild"] = maxtasksperchild  # pylint: disable=E1101
            else:
                self._backend_kwargs["maxtasksperchild"] = maxtasksperchild  # pylint: disable=E1101


def datetime_groupby_apply(
    df, apply_func: Union[Callable, Text], axis=0, level="datetime", resample_rule="ME", n_jobs=-1
):
    """datetime_groupby_apply
    此函数将在datetime级别的索引上应用`apply_func`

    参数
    ----------
    df :
        待处理的DataFrame
    apply_func : Union[Callable, Text]
        处理数据的函数
        如果是字符串则视为pandas原生函数
    axis :
        datetime级别所在的轴
    level :
        datetime级别的名称
    resample_rule :
        用于并行计算的数据重采样规则
    n_jobs :
        joblib的并行任务数
    返回值:
        pd.DataFrame
    """

    def _naive_group_apply(df):
        if isinstance(apply_func, str):
            return getattr(df.groupby(axis=axis, level=level, group_keys=False), apply_func)()
        return df.groupby(level=level, group_keys=False).apply(apply_func)

    if n_jobs != 1:
        dfs = ParallelExt(n_jobs=n_jobs)(
            delayed(_naive_group_apply)(sub_df) for idx, sub_df in df.resample(resample_rule, level=level)
        )
        return pd.concat(dfs, axis=axis).sort_index()
    else:
        return _naive_group_apply(df)


class AsyncCaller:
    """
    AsyncCaller旨在简化异步调用

    目前用于MLflowRecorder中使`log_params`等函数异步执行

    注意:
    - 此调用器不考虑返回值
    """

    STOP_MARK = "__STOP"

    def __init__(self) -> None:
        self._q = Queue()
        self._stop = False
        self._t = Thread(target=self.run)
        self._t.start()

    def close(self):
        self._q.put(self.STOP_MARK)

    def run(self):
        while True:
            # NOTE:
            # atexit will only trigger when all the threads ended. So it may results in deadlock.
            # So the child-threading should actively watch the status of main threading to stop itself.
            main_thread = threading.main_thread()
            if not main_thread.is_alive():
                break
            try:
                data = self._q.get(timeout=1)
            except Empty:
                # NOTE: avoid deadlock. make checking main thread possible
                continue
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        if close:
            self.close()
        self._t.join()

    @staticmethod
    def async_dec(ac_attr):
        def decorator_func(func):
            def wrapper(self, *args, **kwargs):
                if isinstance(getattr(self, ac_attr, None), Callable):
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


# # Outlines: Joblib enhancement
# The code are for implementing following workflow
# - Construct complex data structure nested with delayed joblib tasks
#      - For example,  {"job": [<delayed_joblib_task>,  {"1": <delayed_joblib_task>}]}
# - executing all the tasks and replace all the <delayed_joblib_task> with its return value

# This will make it easier to convert some existing code to a parallel one


class DelayedTask:
    def get_delayed_tuple(self):
        """get_delayed_tuple.
        Return the delayed_tuple created by joblib.delayed
        """
        raise NotImplementedError("NotImplemented")

    def set_res(self, res):
        """set_res.

        Parameters
        ----------
        res :
            the executed result of the delayed tuple
        """
        self.res = res

    def get_replacement(self):
        """return the object to replace the delayed task"""
        raise NotImplementedError("NotImplemented")


class DelayedTuple(DelayedTask):
    def __init__(self, delayed_tpl):
        self.delayed_tpl = delayed_tpl
        self.res = None

    def get_delayed_tuple(self):
        return self.delayed_tpl

    def get_replacement(self):
        return self.res


class DelayedDict(DelayedTask):
    """DelayedDict
    设计用于以下特性:
    将现有代码转换为并行:
    - 构造字典
    - 可立即获取键
    - 值计算耗时
        - 且所有值都在单个函数中计算
    """

    def __init__(self, key_l, delayed_tpl):
        self.key_l = key_l
        self.delayed_tpl = delayed_tpl

    def get_delayed_tuple(self):
        return self.delayed_tpl

    def get_replacement(self):
        return dict(zip(self.key_l, self.res))


def is_delayed_tuple(obj) -> bool:
    """is_delayed_tuple

    参数
    ----------
    obj : object
        待判断对象

    返回值
    -------
    bool
        判断`obj`是否为joblib.delayed元组
    """
    return isinstance(obj, tuple) and len(obj) == 3 and callable(obj[0])


def _replace_and_get_dt(complex_iter):
    """_replace_and_get_dt

    FIXME: 当复杂数据结构包含循环引用时，此函数可能导致无限循环

    参数
    ----------
    complex_iter :
        复杂迭代对象
    """
    if isinstance(complex_iter, DelayedTask):
        dt = complex_iter
        return dt, [dt]
    elif is_delayed_tuple(complex_iter):
        dt = DelayedTuple(complex_iter)
        return dt, [dt]
    elif isinstance(complex_iter, (list, tuple)):
        new_ci = []
        dt_all = []
        for item in complex_iter:
            new_item, dt_list = _replace_and_get_dt(item)
            new_ci.append(new_item)
            dt_all += dt_list
        return new_ci, dt_all
    elif isinstance(complex_iter, dict):
        new_ci = {}
        dt_all = []
        for key, item in complex_iter.items():
            new_item, dt_list = _replace_and_get_dt(item)
            new_ci[key] = new_item
            dt_all += dt_list
        return new_ci, dt_all
    else:
        return complex_iter, []


def _recover_dt(complex_iter):
    """_recover_dt

    将`complex_iter`中所有DelayedTask替换为其`.res`值

    FIXME: 当复杂数据结构包含循环引用时，此函数可能导致无限循环

    参数
    ----------
    complex_iter :
        复杂迭代对象
    """
    if isinstance(complex_iter, DelayedTask):
        return complex_iter.get_replacement()
    elif isinstance(complex_iter, (list, tuple)):
        return [_recover_dt(item) for item in complex_iter]
    elif isinstance(complex_iter, dict):
        return {key: _recover_dt(item) for key, item in complex_iter.items()}
    else:
        return complex_iter


def complex_parallel(paral: Parallel, complex_iter):
    """complex_parallel
    查找complex_iter中由delayed创建的延迟函数，并行运行它们并用结果替换

    >>> from qlib.utils.paral import complex_parallel
    >>> from joblib import Parallel, delayed
    >>> complex_iter = {"a": delayed(sum)([1,2,3]), "b": [1, 2, delayed(sum)([10, 1])]}
    >>> complex_parallel(Parallel(), complex_iter)
    {'a': 6, 'b': [1, 2, 11]}

    参数
    ----------
    paral : Parallel
        并行执行器
    complex_iter :
        注意: 仅会探索list、tuple和dict类型

    返回值
    -------
    complex_iter中延迟的joblib任务被替换为其执行结果
    """

    complex_iter, dt_all = _replace_and_get_dt(complex_iter)
    for res, dt in zip(paral(dt.get_delayed_tuple() for dt in dt_all), dt_all):
        dt.set_res(res)
    complex_iter = _recover_dt(complex_iter)
    return complex_iter


class call_in_subproc:
    """
    当重复运行函数时，很难避免内存泄漏。
    因此在子进程中运行以确保安全。

    注意: 由于本地对象不能被pickle，无法通过闭包实现，
          必须通过可调用类实现。
    """

    def __init__(self, func: Callable, qlib_config: QlibConfig = None):
        """
        参数
        ----------
        func : Callable
            待包装的函数

        qlib_config : QlibConfig
            子进程中初始化Qlib的配置

        返回值
        -------
        Callable
        """
        self.func = func
        self.qlib_config = qlib_config

    def _func_mod(self, *args, **kwargs):
        """通过添加Qlib初始化来修改初始函数"""
        if self.qlib_config is not None:
            C.register_from_C(self.qlib_config)
        return self.func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            return executor.submit(self._func_mod, *args, **kwargs).result()
