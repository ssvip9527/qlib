# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import tempfile
import contextlib
from typing import Optional, Text, IO, Union
from pathlib import Path

from qlib.log import get_module_logger

log = get_module_logger("utils.file")


def get_or_create_path(path: Optional[Text] = None, return_dir: bool = False):
    """根据路径和return_dir参数创建或获取文件/目录

    参数
    ----------
    path: 表示路径的字符串，None表示创建临时路径
    return_dir: 如果为True，创建并返回目录；否则创建并返回文件

    """
    if path:
        if return_dir and not os.path.exists(path):
            os.makedirs(path)
        elif not return_dir:  # return a file, thus we need to create its parent directory
            xpath = os.path.abspath(os.path.join(path, ".."))
            if not os.path.exists(xpath):
                os.makedirs(xpath)
    else:
        temp_dir = tempfile.gettempdir()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if return_dir:
            _, path = tempfile.mkdtemp(dir=temp_dir)
        else:
            _, path = tempfile.mkstemp(dir=temp_dir)
    return path


@contextlib.contextmanager
def save_multiple_parts_file(filename, format="gztar"):
    """保存多部分文件

    实现流程:
        1. 获取'filename'的绝对路径
        2. 创建'filename'目录
        3. 用户对file_path('filename/')进行操作
        4. 删除'filename'目录
        5. 将'filename'目录打包，并将归档文件重命名为filename

    :param filename: 结果模型路径
    :param format: 归档格式: 可选"zip"、"tar"、"gztar"、"bztar"或"xztar"
    :return: 实际模型路径

    用法::

        >>> # 以下代码将创建一个包含'test_doc_i'(i为0-10)文件的归档文件('~/tmp/test_file')
        >>> with save_multiple_parts_file('~/tmp/test_file') as filename_dir:
        ...   for i in range(10):
        ...       temp_path = os.path.join(filename_dir, 'test_doc_{}'.format(str(i)))
        ...       with open(temp_path) as fp:
        ...           fp.write(str(i))
        ...

    """

    if filename.startswith("~"):
        filename = os.path.expanduser(filename)

    file_path = os.path.abspath(filename)

    # Create model dir
    if os.path.exists(file_path):
        raise FileExistsError("ERROR: file exists: {}, cannot be create the directory.".format(file_path))

    os.makedirs(file_path)

    # return model dir
    yield file_path

    # filename dir to filename.tar.gz file
    tar_file = shutil.make_archive(file_path, format=format, root_dir=file_path)

    # Remove filename dir
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    # filename.tar.gz rename to filename
    os.rename(tar_file, file_path)


@contextlib.contextmanager
def unpack_archive_with_buffer(buffer, format="gztar"):
    """使用归档缓冲区解压文件
    调用完成后，归档文件和目录将被删除

    实现流程:
        1. 在'~/tmp/'创建临时文件和目录
        2. 将'buffer'写入临时文件
        3. 解压归档文件('tempfile')
        4. 用户对file_path('tempfile/')进行操作
        5. 删除临时文件和临时文件目录

    :param buffer: 字节数据
    :param format: 归档格式: 可选"zip"、"tar"、"gztar"、"bztar"或"xztar"
    :return: 解压后的归档目录路径

    用法::

        >>> # 以下代码打印'test_unpack.tar.gz'中的所有文件名
        >>> with open('test_unpack.tar.gz') as fp:
        ...     buffer = fp.read()
        ...
        >>> with unpack_archive_with_buffer(buffer) as temp_dir:
        ...     for f_n in os.listdir(temp_dir):
        ...         print(f_n)
        ...

    """
    temp_dir = tempfile.gettempdir()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name

    try:
        tar_file = file_path + ".tar.gz"
        os.rename(file_path, tar_file)
        # Create dir
        os.makedirs(file_path)
        shutil.unpack_archive(tar_file, format=format, extract_dir=file_path)

        # Return temp dir
        yield file_path

    except Exception as e:
        log.error(str(e))
    finally:
        # Remove temp tar file
        if os.path.exists(tar_file):
            os.unlink(tar_file)

        # Remove temp model dir
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


@contextlib.contextmanager
def get_tmp_file_with_buffer(buffer):
    temp_dir = tempfile.gettempdir()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=True, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name
        yield file_path


@contextlib.contextmanager
def get_io_object(file: Union[IO, str, Path], *args, **kwargs) -> IO:
    """
    提供获取IO对象的简易接口

    参数
    ----------
    file : Union[IO, str, Path]
        表示文件的对象

    返回
    -------
    IO:
        类IO对象

    异常
    ------
        NotImplementedError:
    """
    if isinstance(file, IO):
        yield file
    else:
        if isinstance(file, str):
            file = Path(file)
        if not isinstance(file, Path):
            raise NotImplementedError(f"This type[{type(file)}] of input is not supported")
        with file.open(*args, **kwargs) as f:
            yield f
