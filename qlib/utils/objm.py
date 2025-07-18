# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import pickle
import tempfile
from pathlib import Path

from qlib.config import C


class ObjManager:
    def save_obj(self, obj: object, name: str):
        """
        保存对象

        参数
        ----------
        obj : object
            要保存的对象
        name : str
            对象名称
        """
        raise NotImplementedError(f"Please implement `save_obj`")

    def save_objs(self, obj_name_l):
        """
        保存多个对象

        参数
        ----------
        obj_name_l : list of <obj, name>
            对象和名称的列表
        """
        raise NotImplementedError(f"Please implement the `save_objs` method")

    def load_obj(self, name: str) -> object:
        """
        根据名称加载对象

        参数
        ----------
        name : str
            对象名称

        返回值
        -------
        object:
            加载的对象
        """
        raise NotImplementedError(f"Please implement the `load_obj` method")

    def exists(self, name: str) -> bool:
        """
        检查指定名称的对象是否存在

        参数
        ----------
        name : str
            对象名称

        返回值
        -------
        bool:
            对象是否存在
        """
        raise NotImplementedError(f"Please implement the `exists` method")

    def list(self) -> list:
        """
        列出所有对象

        返回值
        -------
        list:
            对象列表
        """
        raise NotImplementedError(f"Please implement the `list` method")

    def remove(self, fname=None):
        """删除对象

        参数
        ----------
        fname :
            如果提供文件名，则删除特定文件
            否则将删除所有对象
        """
        raise NotImplementedError(f"Please implement the `remove` method")


class FileManager(ObjManager):
    """
    使用文件系统管理对象
    """

    def __init__(self, path=None):
        if path is None:
            self.path = Path(self.create_path())
        else:
            self.path = Path(path).resolve()

    def create_path(self) -> str:
        try:
            return tempfile.mkdtemp(prefix=str(C["file_manager_path"]) + os.sep)
        except AttributeError as attribute_e:
            raise NotImplementedError(
                f"如果未提供路径，则应实现`create_path`函数"
            ) from attribute_e

    def save_obj(self, obj, name):
        with (self.path / name).open("wb") as f:
            pickle.dump(obj, f, protocol=C.dump_protocol_version)

    def save_objs(self, obj_name_l):
        for obj, name in obj_name_l:
            self.save_obj(obj, name)

    def load_obj(self, name):
        with (self.path / name).open("rb") as f:
            return pickle.load(f)

    def exists(self, name):
        return (self.path / name).exists()

    def list(self):
        return list(self.path.iterdir())

    def remove(self, fname=None):
        if fname is None:
            for fp in self.path.glob("*"):
                fp.unlink()
            self.path.rmdir()
        else:
            (self.path / fname).unlink()
