# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
所有与模块相关的类，例如：
- 导入模块、类
- 遍历模块
- 对类或模块的操作...
"""

import contextlib
import importlib
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse

from qlib.typehint import InstConf


def get_module_by_module_path(module_path: Union[str, ModuleType]):
    """加载模块路径

    :param module_path: 模块路径
    :return: 模块对象
    :raises: ModuleNotFoundError 当模块不存在时抛出
    """
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def split_module_path(module_path: str) -> Tuple[str, str]:
    """
    参数
    ----------
    module_path : str
        例如: "a.b.c.ClassName"

    返回值
    -------
    Tuple[str, str]
        例如: ("a.b.c", "ClassName")
    """
    *m_path, cls = module_path.split(".")
    m_path = ".".join(m_path)
    return m_path, cls


def get_callable_kwargs(config: InstConf, default_module: Union[str, ModuleType] = None) -> (type, dict):
    """
    从配置信息中提取类/函数及其参数

    参数
    ----------
    config : [dict, str]
        类似配置信息
        请参考init_instance_by_config的文档

    default_module : Python模块或str
        应该是一个Python模块用于加载类类型
        此函数会首先从config['module_path']加载类
        如果config['module_path']不存在，则从default_module加载类

    返回值
    -------
    (type, dict):
        类/函数对象及其参数

    Raises
    ------
        ModuleNotFoundError
    """
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            # 1) get module and class
            # - case 1): "a.b.c.ClassName"
            # - case 2): {"class": "ClassName", "module_path": "a.b.c"}
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            module = get_module_by_module_path(m_path)

            # 2) get callable
            _callable = getattr(module, cls)  # may raise AttributeError
        else:
            _callable = config[key]  # the class type itself is passed in
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        # a.b.c.ClassName
        m_path, cls = split_module_path(config)
        module = get_module_by_module_path(default_module if m_path == "" else m_path)

        _callable = getattr(module, cls)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return _callable, kwargs


get_cls_kwargs = get_callable_kwargs  # NOTE: this is for compatibility for the previous version


def init_instance_by_config(
    config: InstConf,
    default_module=None,
    accept_types: Union[type, Tuple[type]] = (),
    try_kwargs: Dict = {},
    **kwargs,
) -> Any:
    """
    通过配置获取初始化实例

    参数
    ----------
    config : InstConf

    default_module : Python模块
        可选。应该是一个Python模块。
        注意："module_path"会被`module`参数覆盖

        此函数会首先从config['module_path']加载类
        如果config['module_path']不存在，则从default_module加载类

    accept_types: Union[type, Tuple[type]]
        可选。如果配置是特定类型的实例，则直接返回配置
        这将传递给isinstance的第二个参数

    try_kwargs: Dict
        尝试在初始化实例时传入`try_kwargs`中的参数
        如果出错，将回退到不使用try_kwargs的初始化

    返回值
    -------
    object:
        基于配置信息初始化的对象
    """
    if isinstance(config, accept_types):
        return config

    if isinstance(config, (str, Path)):
        if isinstance(config, str):
            # path like 'file:///<path to pickle file>/obj.pkl'
            pr = urlparse(config)
            if pr.scheme == "file":

                # To enable relative path like file://data/a/b/c.pkl.  pr.netloc will be data
                path = pr.path
                if pr.netloc != "":
                    path = path.lstrip("/")

                pr_path = os.path.join(pr.netloc, path) if bool(pr.path) else pr.netloc
                with open(os.path.normpath(pr_path), "rb") as f:
                    return pickle.load(f)
        else:
            with config.open("rb") as f:
                return pickle.load(f)

    klass, cls_kwargs = get_callable_kwargs(config, default_module=default_module)

    try:
        return klass(**cls_kwargs, **try_kwargs, **kwargs)
    except (TypeError,):
        # TypeError for handling errors like
        # 1: `XXX() got multiple values for keyword argument 'YYY'`
        # 2: `XXX() got an unexpected keyword argument 'YYY'
        return klass(**cls_kwargs, **kwargs)


@contextlib.contextmanager
def class_casting(obj: object, cls: type):
    """
    Python不提供向下转型机制
    我们在这里使用技巧来实现类向下转型

    参数
    ----------
    obj : object
        要转型的对象
    cls : type
        目标类类型
    """
    orig_cls = obj.__class__
    obj.__class__ = cls
    yield
    obj.__class__ = orig_cls


def find_all_classes(module_path: Union[str, ModuleType], cls: type) -> List[type]:
    """
    递归查找给定模块中继承自`cls`的所有类
    - `cls`本身也会包含在内

        >>> from qlib.data.dataset.handler import DataHandler
        >>> find_all_classes("qlib.contrib.data.handler", DataHandler)
        [<class 'qlib.contrib.data.handler.Alpha158'>, <class 'qlib.contrib.data.handler.Alpha158vwap'>, <class 'qlib.contrib.data.handler.Alpha360'>, <class 'qlib.contrib.data.handler.Alpha360vwap'>, <class 'qlib.data.dataset.handler.DataHandlerLP'>]

    待办:
    - 跳过导入错误

    """
    if isinstance(module_path, ModuleType):
        mod = module_path
    else:
        mod = importlib.import_module(module_path)

    cls_list = []

    def _append_cls(obj):
        # Leverage the closure trick to reuse code
        if isinstance(obj, type) and issubclass(obj, cls) and cls not in cls_list:
            cls_list.append(obj)

    for attr in dir(mod):
        _append_cls(getattr(mod, attr))

    if hasattr(mod, "__path__"):
        # if the model is a package
        for _, modname, _ in pkgutil.iter_modules(mod.__path__):
            sub_mod = importlib.import_module(f"{mod.__package__}.{modname}")
            for m_cls in find_all_classes(sub_mod, cls):
                _append_cls(m_cls)
    return cls_list
