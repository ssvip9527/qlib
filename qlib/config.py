# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
关于配置
=================

配置将基于_default_config。
支持两种模式：
- 客户端
- 服务器

"""
from __future__ import annotations

import os
import re
import copy
import logging
import platform
import multiprocessing
from pathlib import Path
from typing import Callable, Optional, Union
from typing import TYPE_CHECKING

from qlib.constant import REG_CN, REG_US, REG_TW

if TYPE_CHECKING:
    from qlib.utils.time import Freq

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowSettings(BaseSettings):
    uri: str = "file:" + str(Path(os.getcwd()).resolve() / "mlruns")
    default_exp_name: str = "Experiment"


class QSettings(BaseSettings):
    """
    Qlib的设置。
    它尝试为Qlib的大多数组件提供默认设置。
    但为所有Qlib组件提供全面的设置将是一个漫长的过程。

    以下是一些设计指南：
    - 设置的优先级为：
        - 主动传入的设置，例如 `qlib.init(provider_uri=...)`
        - 默认设置
            - QSettings尝试为Qlib的大多数组件提供默认设置。
    """

    mlflow: MLflowSettings = MLflowSettings()

    model_config = SettingsConfigDict(
        env_prefix="QLIB_",
        env_nested_delimiter="_",
    )


QSETTINGS = QSettings()


class Config:
    def __init__(self, default_conf):
        self.__dict__["_default_config"] = copy.deepcopy(default_conf)  # 避免与__getattr__冲突
        self.reset()

    def __getitem__(self, key):
        return self.__dict__["_config"][key]

    def __getattr__(self, attr):
        if attr in self.__dict__["_config"]:
            return self.__dict__["_config"][attr]

        raise AttributeError(f"No such `{attr}` in self._config")

    def get(self, key, default=None):
        return self.__dict__["_config"].get(key, default)

    def __setitem__(self, key, value):
        self.__dict__["_config"][key] = value

    def __setattr__(self, attr, value):
        self.__dict__["_config"][attr] = value

    def __contains__(self, item):
        return item in self.__dict__["_config"]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_config"])

    def __repr__(self):
        return str(self.__dict__["_config"])

    def reset(self):
        self.__dict__["_config"] = copy.deepcopy(self._default_config)

    def update(self, *args, **kwargs):
        self.__dict__["_config"].update(*args, **kwargs)

    def set_conf_from_C(self, config_c):
        self.update(**config_c.__dict__["_config"])

    @staticmethod
    def register_from_C(config, skip_register=True):
        from .utils import set_log_with_config  # pylint: disable=C0415

        if C.registered and skip_register:
            return

        C.set_conf_from_C(config)
        if C.logging_config:
            set_log_with_config(C.logging_config)
        C.register()


# pickle.dump协议版本: https://docs.python.org/3/library/pickle.html#data-stream-format
PROTOCOL_VERSION = 4

NUM_USABLE_CPU = max(multiprocessing.cpu_count() - 2, 1)

DISK_DATASET_CACHE = "DiskDatasetCache"
SIMPLE_DATASET_CACHE = "SimpleDatasetCache"
DISK_EXPRESSION_CACHE = "DiskExpressionCache"

DEPENDENCY_REDIS_CACHE = (DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE)

_default_config = {
    # 数据提供器配置
    "calendar_provider": "LocalCalendarProvider",
    "instrument_provider": "LocalInstrumentProvider",
    "feature_provider": "LocalFeatureProvider",
    "pit_provider": "LocalPITProvider",
    "expression_provider": "LocalExpressionProvider",
    "dataset_provider": "LocalDatasetProvider",
    "provider": "LocalProvider",
    # 在qlib.init()中配置
    # "provider_uri" str or dict:
    #   # str
    #   "~/.qlib/stock_data/cn_data"
    #   # dict
    #   {"day": "~/.qlib/stock_data/cn_data", "1min": "~/.qlib/stock_data/cn_data_1min"}
    # 注意：provider_uri优先级：
    #   1. backend_config: backend_obj["kwargs"]["provider_uri"]
    #   2. backend_config: backend_obj["kwargs"]["provider_uri_map"]
    #   3. qlib.init: provider_uri
    "provider_uri": "",
    # 缓存
    "expression_cache": None,
    "calendar_cache": None,
    # 用于简单数据集缓存
    "local_cache_path": None,
    # kernels can be a fixed value or a callable function lie `def (freq: str) -> int`
    # 如果内核是arctic_kernels，`min(NUM_USABLE_CPU, 30)`可能是一个不错的值
    "kernels": NUM_USABLE_CPU,
    # pickle.dump protocol version
    "dump_protocol_version": PROTOCOL_VERSION,
    # How many tasks belong to one process. Recommend 1 for high-frequency data and None for daily data.
    "maxtasksperchild": None,
    # If joblib_backend is None, use loky
    "joblib_backend": "multiprocessing",
    "default_disk_cache": 1,  # 0:skip/1:use
    "mem_cache_size_limit": 500,
    "mem_cache_limit_type": "length",
    # memory cache expire second, only in used 'DatasetURICache' and 'client D.calendar'
    # default 1 hour
    "mem_cache_expire": 60 * 60,
    # cache dir name
    "dataset_cache_dir_name": "dataset_cache",
    "features_cache_dir_name": "features_cache",
    # redis
    # in order to use cache
    "redis_host": "127.0.0.1",
    "redis_port": 6379,
    "redis_task_db": 1,
    "redis_password": None,
    # This value can be reset via qlib.init
    "logging_level": logging.INFO,
    # Global configuration of qlib log
    # logging_level can control the logging level more finely
    "logging_config": {
        "version": 1,
        "formatters": {
            "logger_format": {
                "format": "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
            }
        },
        "filters": {
            "field_not_found": {
                "()": "qlib.log.LogFilter",
                "param": [".*?WARN: data not found for.*?"],
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": logging.DEBUG,
                "formatter": "logger_format",
                "filters": ["field_not_found"],
            }
        },
        # 通常应设置为`False`以避免重复日志记录 [1]。
        # 但是，由于pytest中的bug，需要将日志消息传播到根日志器才能被`caplog`捕获 [2]。
        # [1] https://github.com/microsoft/qlib/pull/1661
        # [2] https://github.com/pytest-dev/pytest/issues/3697
        "loggers": {"qlib": {"level": logging.DEBUG, "handlers": ["console"], "propagate": False}},
        # 为了让qlib与其他包兼容，我们不应禁用现有日志器。
        # 注意，根据logging文档，此参数默认值为True。
        "disable_existing_loggers": False,
    },
    # 实验管理器的默认配置
    "exp_manager": {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": QSETTINGS.mlflow.uri,
            "default_exp_name": QSETTINGS.mlflow.default_exp_name,
        },
    },
    "pit_record_type": {
        "date": "I",  # uint32
        "period": "I",  # uint32
        "value": "d",  # float64
        "index": "I",  # uint32
    },
    "pit_record_nan": {
        "date": 0,
        "period": 0,
        "value": float("NAN"),
        "index": 0xFFFFFFFF,
    },
    # MongoDB的默认配置
    "mongo": {
        "task_url": "mongodb://localhost:27017/",
        "task_db_name": "default_task_db",
    },
    # 高频分钟数据的偏移分钟数，用于回测
    # 若min_data_shift == 0，使用默认交易时间 [9:30, 11:29, 1:00, 2:59]
    # 若min_data_shift != 0，使用偏移后的交易时间 [9:30, 11:29, 1:00, 2:59] - shift*分钟
    "min_data_shift": 0,
}

MODE_CONF = {
    "server": {
        # 在qlib.init()中配置
        "provider_uri": "",
        # Redis配置
        "redis_host": "127.0.0.1",
        "redis_port": 6379,
        "redis_task_db": 1,
        # 缓存配置
        "expression_cache": DISK_EXPRESSION_CACHE,
        "dataset_cache": DISK_DATASET_CACHE,
        "local_cache_path": Path("~/.cache/qlib_simple_cache").expanduser().resolve(),
        "mount_path": None,
    },
    "client": {
        # 在用户自己的代码中配置
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        # cache
        # 使用参数'remote'表示客户端正在使用server_cache，此时写入权限将被禁用。
        # 默认禁用缓存。避免为初学者引入高级功能
        "dataset_cache": None,
        # SimpleDatasetCache目录
        "local_cache_path": Path("~/.cache/qlib_simple_cache").expanduser().resolve(),
        # 客户端配置
        "mount_path": None,
        "auto_mount": False,  # NFS已在我们的服务器上挂载[auto_mount: False]。
        # 在其他服务器（如PAI）上，NFS应由qlib自动挂载[auto_mount:True]
        "timeout": 100,
        "logging_level": logging.INFO,
        "region": REG_CN,
        # 自定义算子
        # custom_ops的每个元素应为Type[ExpressionOps]或dict
        # 若custom_ops的元素为Type[ExpressionOps]，表示自定义算子类
        # 若custom_ops的元素为dict，表示自定义算子的配置，应包含`class`和`module_path`键
        "custom_ops": [],
    },
}

HIGH_FREQ_CONFIG = {
    "provider_uri": "~/.qlib/qlib_data/cn_data_1min",
    "dataset_cache": None,
    "expression_cache": "DiskExpressionCache",
    "region": REG_CN,
}

_default_region_config = {
    REG_CN: {
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": "close",
    },
    REG_US: {
        "trade_unit": 1,
        "limit_threshold": None,
        "deal_price": "close",
    },
    REG_TW: {
        "trade_unit": 1000,
        "limit_threshold": 0.1,
        "deal_price": "close",
    },
}


class QlibConfig(Config):
    # URI_TYPE
    LOCAL_URI = "local"
    NFS_URI = "nfs"
    DEFAULT_FREQ = "__DEFAULT_FREQ"

    def __init__(self, default_conf):
        super().__init__(default_conf)
        self._registered = False

    class DataPathManager:
        """
        动机：
        - 根据给定信息（如provider_uri、mount_path和频率）获取访问数据的正确路径（例如数据URI）
        - 一些用于处理URI的辅助函数。
        """

        def __init__(self, provider_uri: Union[str, Path, dict], mount_path: Union[str, Path, dict]):
            """
            `provider_uri`和`mount_path`的关系
            - 仅当provider_uri是NFS路径时才使用`mount_path`
            - 否则，provider_uri将用于访问数据
            """
            self.provider_uri = provider_uri
            self.mount_path = mount_path

        @staticmethod
        def format_provider_uri(provider_uri: Union[str, dict, Path]) -> dict:
            if provider_uri is None:
                raise ValueError("provider_uri不能为空")
            if isinstance(provider_uri, (str, dict, Path)):
                if not isinstance(provider_uri, dict):
                    provider_uri = {QlibConfig.DEFAULT_FREQ: provider_uri}
            else:
                raise TypeError(f"provider_uri不支持{type(provider_uri)}类型")
            for freq, _uri in provider_uri.items():
                if QlibConfig.DataPathManager.get_uri_type(_uri) == QlibConfig.LOCAL_URI:
                    provider_uri[freq] = str(Path(_uri).expanduser().resolve())
            return provider_uri

        @staticmethod
        def get_uri_type(uri: Union[str, Path]):
            uri = uri if isinstance(uri, str) else str(uri.expanduser().resolve())
            is_win = re.match("^[a-zA-Z]:.*", uri) is not None  # 例如'C:\\data', 'D:'
            # 例如'host:/data/'（用户可自行定义短主机名或使用localhost）
            is_nfs_or_win = re.match("^[^/]+:.+", uri) is not None

            if is_nfs_or_win and not is_win:
                return QlibConfig.NFS_URI
            else:
                return QlibConfig.LOCAL_URI

        def get_data_uri(self, freq: Optional[Union[str, Freq]] = None) -> Path:
            """
            请参考DataPathManager的__init__和类文档
            """
            if freq is not None:
                freq = str(freq)  # converting Freq to string
            if freq is None or freq not in self.provider_uri:
                freq = QlibConfig.DEFAULT_FREQ
            _provider_uri = self.provider_uri[freq]
            if self.get_uri_type(_provider_uri) == QlibConfig.LOCAL_URI:
                return Path(_provider_uri)
            elif self.get_uri_type(_provider_uri) == QlibConfig.NFS_URI:
                if "win" in platform.system().lower():
                    # windows, mount_path is the drive
                    _path = str(self.mount_path[freq])
                    return Path(f"{_path}:\\") if ":" not in _path else Path(_path)
                return Path(self.mount_path[freq])
            else:
                raise NotImplementedError(f"This type of uri is not supported")

    def set_mode(self, mode):
        # raise KeyError
        self.update(MODE_CONF[mode])
        # TODO: 根据kwargs更新地区

    def set_region(self, region):
        # raise KeyError
        self.update(_default_region_config[region])

    @staticmethod
    def is_depend_redis(cache_name: str):
        return cache_name in DEPENDENCY_REDIS_CACHE

    @property
    def dpm(self):
        return self.DataPathManager(self["provider_uri"], self["mount_path"])

    def resolve_path(self):
        # 解析路径
        _mount_path = self["mount_path"]
        _provider_uri = self.DataPathManager.format_provider_uri(self["provider_uri"])
        if not isinstance(_mount_path, dict):
            _mount_path = {_freq: _mount_path for _freq in _provider_uri.keys()}

        # 检查provider_uri和mount_path
        _miss_freq = set(_provider_uri.keys()) - set(_mount_path.keys())
        assert len(_miss_freq) == 0, f"mount_path is missing freq: {_miss_freq}"

        # 解析处理
        for _freq in _provider_uri.keys():
            # mount_path
            _mount_path[_freq] = (
                _mount_path[_freq] if _mount_path[_freq] is None else str(Path(_mount_path[_freq]).expanduser())
            )
        self["provider_uri"] = _provider_uri
        self["mount_path"] = _mount_path

    def set(self, default_conf: str = "client", **kwargs):
        """
        根据输入参数配置qlib

        配置将像字典一样工作。

        通常，它会根据键直接替换值。
        但是，当配置嵌套且复杂时，用户有时难以设置配置

        因此，此API提供了一些特殊参数，方便用户更便捷地设置键。
        - region: REG_CN, REG_US
            - 多个与地区相关的配置将被更改

        参数
        ----------
        default_conf : str
            用户选择的默认配置模板："server"（服务器）, "client"（客户端）
        """
        from .utils import set_log_with_config, get_module_logger, can_use_cache  # pylint: disable=C0415

        self.reset()

        _logging_config = kwargs.get("logging_config", self.logging_config)

        # 设置全局配置
        if _logging_config:
            set_log_with_config(_logging_config)

        logger = get_module_logger("Initialization", kwargs.get("logging_level", self.logging_level))
        logger.info(f"default_conf: {default_conf}.")

        self.set_mode(default_conf)
        self.set_region(kwargs.get("region", self["region"] if "region" in self else REG_CN))

        for k, v in kwargs.items():
            if k not in self:
                logger.warning("Unrecognized config %s" % k)
            self[k] = v

        self.resolve_path()

        if not (self["expression_cache"] is None and self["dataset_cache"] is None):
            # 检查Redis
            if not can_use_cache():
                log_str = ""
                # 检查表达式缓存
                if self.is_depend_redis(self["expression_cache"]):
                    log_str += self["expression_cache"]
                    self["expression_cache"] = None
                # 检查数据集缓存
                if self.is_depend_redis(self["dataset_cache"]):
                    log_str += f" and {self['dataset_cache']}" if log_str else self["dataset_cache"]
                    self["dataset_cache"] = None
                if log_str:
                    logger.warning(
                        f"redis connection failed(host={self['redis_host']} port={self['redis_port']}), "
                        f"{log_str} will not be used!"
                    )

    def register(self):
        from .utils import init_instance_by_config  # pylint: disable=C0415
        from .data.ops import register_all_ops  # pylint: disable=C0415
        from .data.data import register_all_wrappers  # pylint: disable=C0415
        from .workflow import R, QlibRecorder  # pylint: disable=C0415
        from .workflow.utils import experiment_exit_handler  # pylint: disable=C0415

        register_all_ops(self)
        register_all_wrappers(self)
        # set up QlibRecorder
        exp_manager = init_instance_by_config(self["exp_manager"])
        qr = QlibRecorder(exp_manager)
        R.register(qr)
        # Python程序结束时清理实验
        experiment_exit_handler()

        # 支持用户重置qlib版本（当用户希望使用旧版本连接到qlib服务器时有用）
        self.reset_qlib_version()

        self._registered = True

    def reset_qlib_version(self):
        import qlib  # pylint: disable=C0415

        reset_version = self.get("qlib_reset_version", None)
        if reset_version is not None:
            qlib.__version__ = reset_version
        else:
            qlib.__version__ = getattr(qlib, "__version__bak")
            # Due to a bug? that converting __version__ to _QlibConfig__version__bak
            # Using  __version__bak instead of __version__

    def get_kernels(self, freq: str):
        """根据频率获取处理器数量"""
        if isinstance(self["kernels"], Callable):
            return self["kernels"](freq)
        return self["kernels"]

    @property
    def registered(self):
        return self._registered


# global config
C = QlibConfig(_default_config)
