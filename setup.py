from setuptools import setup, Extension
import numpy
import os


# 读取指定相对路径文件内容
def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


# 从指定文件中获取版本号
def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("无法找到版本信息。")


# 获取 numpy 的头文件路径
NUMPY_INCLUDE = numpy.get_include()

# 获取 qlib 包的版本号
VERSION = get_version("qlib/__init__.py")


# 配置 Cython 扩展模块并进行安装
setup(
    version=VERSION,
    ext_modules=[
        Extension(
            "qlib.data._libs.rolling",
            ["qlib/data/_libs/rolling.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
        Extension(
            "qlib.data._libs.expanding",
            ["qlib/data/_libs/expanding.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
    ],
)
