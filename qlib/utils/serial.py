# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import dill
from pathlib import Path
from typing import Union
from ..config import C


class Serializable:
    """Serializable类将改变pickle的行为。

        判断属性在dump时是否保留的规则(优先级从高到低):
        - 在config属性列表中 -> 总是丢弃
        - 在include属性列表中 -> 总是保留
        - 在exclude属性列表中 -> 总是丢弃
        - 不以`_`开头的属性名 -> 保留
        - 以`_`开头的属性名 -> 如果`dump_all`为true则保留，否则丢弃

    它提供了一种语法糖来区分用户不希望保存的属性。
    - 例如，一个可学习的Datahandler在dump到磁盘时只想保存参数而不保存数据
    """

    pickle_backend = "pickle"  # another optional value is "dill" which can pickle more things of python.
    default_dump_all = False  # if dump all things
    config_attr = ["_include", "_exclude"]
    exclude_attr = []  # exclude_attr have lower priorities than `self._exclude`
    include_attr = []  # include_attr have lower priorities then `self._include`
    FLAG_KEY = "_qlib_serial_flag"

    def __init__(self):
        self._dump_all = self.default_dump_all
        self._exclude = None  # this attribute have higher priorities than `exclude_attr`

    def _is_kept(self, key):
        if key in self.config_attr:
            return False
        if key in self._get_attr_list("include"):
            return True
        if key in self._get_attr_list("exclude"):
            return False
        return self.dump_all or not key.startswith("_")

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if self._is_kept(k)}

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @property
    def dump_all(self):
        """对象是否dump所有属性
        """
        return getattr(self, "_dump_all", False)

    def _get_attr_list(self, attr_type: str) -> list:
        """获取特定类型的属性列表

        参数
        ----------
        attr_type : str
            "include"(包含)或"exclude"(排除)

        返回值
        -------
        list:
            属性列表
        """
        if hasattr(self, f"_{attr_type}"):
            res = getattr(self, f"_{attr_type}", [])
        else:
            res = getattr(self.__class__, f"{attr_type}_attr", [])
        if res is None:
            return []
        return res

    def config(self, recursive=False, **kwargs):
        """配置可序列化对象

        参数
        ----------
        kwargs可能包含以下键:

            dump_all : bool
                对象是否dump所有属性
            exclude : list
                不被dump的属性列表
            include : list
                被dump的属性列表

        recursive : bool
            是否递归配置
        """
        keys = {"dump_all", "exclude", "include"}
        for k, v in kwargs.items():
            if k in keys:
                attr_name = f"_{k}"
                setattr(self, attr_name, v)
            else:
                raise KeyError(f"Unknown parameter: {k}")

        if recursive:
            for obj in self.__dict__.values():
                # set flag to prevent endless loop
                self.__dict__[self.FLAG_KEY] = True
                if isinstance(obj, Serializable) and self.FLAG_KEY not in obj.__dict__:
                    obj.config(recursive=True, **kwargs)
                del self.__dict__[self.FLAG_KEY]

    def to_pickle(self, path: Union[Path, str], **kwargs):
        """将对象dump到pickle文件

        path (Union[Path, str]): dump文件路径

        kwargs可能包含以下键:

            dump_all : bool
                对象是否dump所有属性
            exclude : list
                不被dump的属性列表
            include : list
                被dump的属性列表
        """
        self.config(**kwargs)
        with Path(path).open("wb") as f:
            # pickle interface like backend; such as dill
            self.get_backend().dump(self, f, protocol=C.dump_protocol_version)

    @classmethod
    def load(cls, filepath):
        """
        从文件路径加载可序列化类

        参数:
            filepath (str): 文件路径

        异常:
            TypeError: pickle文件必须是`type(cls)`类型

        返回:
            `type(cls)`: `type(cls)`的实例
        """
        with open(filepath, "rb") as f:
            object = cls.get_backend().load(f)
        if isinstance(object, cls):
            return object
        else:
            raise TypeError(f"The instance of {type(object)} is not a valid `{type(cls)}`!")

    @classmethod
    def get_backend(cls):
        """返回Serializable类的真实后端。pickle_backend值可以是"pickle"或"dill"

        返回:
            module: 基于pickle_backend的pickle或dill模块
        """
        # NOTE: pickle interface like backend; such as dill
        if cls.pickle_backend == "pickle":
            return pickle
        elif cls.pickle_backend == "dill":
            return dill
        else:
            raise ValueError("Unknown pickle backend, please use 'pickle' or 'dill'.")

    @staticmethod
    def general_dump(obj, path: Union[Path, str]):
        """对象的通用dump方法

        参数
        ----------
        obj : object
            待dump的对象
        path : Union[Path, str]
            数据将被dump的目标路径
        """
        path = Path(path)
        if isinstance(obj, Serializable):
            obj.to_pickle(path)
        else:
            with path.open("wb") as f:
                pickle.dump(obj, f, protocol=C.dump_protocol_version)
