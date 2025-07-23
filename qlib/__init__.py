# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

__version__ = "0.9.7.11"
__version__bak = __version__  # This version is backup for QlibConfig.reset_qlib_version
import os
import re
from typing import Union
from ruamel.yaml import YAML
import logging
import platform
import subprocess
from .log import get_module_logger


# init qlib
def init(default_conf="client", **kwargs):
    """

    参数
    ----------
    default_conf: str
        默认值为client。接受的值：client/server。
    **kwargs :
        clear_mem_cache: bool
            默认值为True；
            是否清除内存缓存。
            当多次调用init时，常用于提高性能
        skip_if_reg: bool
            默认值为True；
            使用记录器时，可将skip_if_reg设为True以避免记录器丢失。

    """
    from .config import C  # pylint: disable=C0415
    from .data.cache import H  # pylint: disable=C0415

    logger = get_module_logger("Initialization")

    skip_if_reg = kwargs.pop("skip_if_reg", False)
    if skip_if_reg and C.registered:
        # 如果在运行实验`R.start`期间重新初始化Qlib。
        # 这将导致记录器丢失
        logger.warning("Skip initialization because `skip_if_reg is True`")
        return

    clear_mem_cache = kwargs.pop("clear_mem_cache", True)
    if clear_mem_cache:
        H.clear()
    C.set(default_conf, **kwargs)
    get_module_logger.setLevel(C.logging_level)

    # mount nfs
    for _freq, provider_uri in C.provider_uri.items():
        mount_path = C["mount_path"][_freq]
        # check path if server/local
        uri_type = C.dpm.get_uri_type(provider_uri)
        if uri_type == C.LOCAL_URI:
            if not Path(provider_uri).exists():
                if C["auto_mount"]:
                    logger.error(
                        f"Invalid provider uri: {provider_uri}, please check if a valid provider uri has been set. This path does not exist."
                    )
                else:
                    logger.warning(f"auto_path is False, please make sure {mount_path} is mounted")
        elif uri_type == C.NFS_URI:
            _mount_nfs_uri(provider_uri, C.dpm.get_data_uri(_freq), C["auto_mount"])
        else:
            raise NotImplementedError(f"不支持此类型的URI")

    C.register()

    if "flask_server" in C:
        logger.info(f"flask_server={C['flask_server']}, flask_port={C['flask_port']}")
    logger.info("qlib已基于%s设置成功初始化。" % default_conf)
    data_path = {_freq: C.dpm.get_data_uri(_freq) for _freq in C.dpm.provider_uri.keys()}
    logger.info(f"data_path={data_path}")


def _mount_nfs_uri(provider_uri, mount_path, auto_mount: bool = False):
    LOG = get_module_logger("mount nfs", level=logging.INFO)
    if mount_path is None:
        raise ValueError(f"无效的挂载路径: {mount_path}!")
    if not re.match(r"^[a-zA-Z0-9.:/\-_]+$", provider_uri):
        raise ValueError(f"无效的provider_uri格式: {provider_uri}")
    # FIXME: C["provider_uri"]在此函数中被修改
    # 如果不修改，我们只需传递provider_uri或mount_path而不是C
    mount_command = ["sudo", "mount.nfs", provider_uri, mount_path]
    # 如果provider uri的格式类似于 172.23.233.89//data/csdesign'
    # 这将被视为nfs路径，将使用客户端提供程序
    if not auto_mount:  # pylint: disable=R1702
        if not Path(mount_path).exists():
            raise FileNotFoundError(
                f"无效的挂载路径: {mount_path}! 请手动挂载: {' '.join(mount_command)} 或设置初始化参数`auto_mount=True`"
            )
    else:
        # Judging system type
        sys_type = platform.system()
        if "windows" in sys_type.lower():
            # system: window
            try:
                subprocess.run(
                    ["mount", "-o", "anon", provider_uri, mount_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                LOG.info("Mount finished.")
            except subprocess.CalledProcessError as e:
                error_output = (e.stdout or "") + (e.stderr or "")
                if e.returncode == 85:
                    LOG.warning(f"{provider_uri} already mounted at {mount_path}")
                elif e.returncode == 53:
                    raise OSError("网络路径未找到") from e
                elif "error" in error_output.lower() or "错误" in error_output:
                    raise OSError("无效的挂载路径") from e
                else:
                    raise OSError(f"未知的挂载错误: {error_output.strip()}") from e
        else:
            # system: linux/Unix/Mac
            # check mount
            _remote_uri = provider_uri[:-1] if provider_uri.endswith("/") else provider_uri
            # `mount a /b/c` is different from `mount a /b/c/`. So we convert it into string to make sure handling it accurately
            mount_path = str(mount_path)
            _mount_path = mount_path[:-1] if mount_path.endswith("/") else mount_path
            _check_level_num = 2
            _is_mount = False
            while _check_level_num:
                with subprocess.Popen(
                    ["mount"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ) as shell_r:
                    _command_log = shell_r.stdout.readlines()
                    _command_log = [line for line in _command_log if _remote_uri in line]
                if len(_command_log) > 0:
                    for _c in _command_log:
                        _temp_mount = _c.decode("utf-8").split(" ")[2]
                        _temp_mount = _temp_mount[:-1] if _temp_mount.endswith("/") else _temp_mount
                        if _temp_mount == _mount_path:
                            _is_mount = True
                            break
                if _is_mount:
                    break
                _remote_uri = "/".join(_remote_uri.split("/")[:-1])
                _mount_path = "/".join(_mount_path.split("/")[:-1])
                _check_level_num -= 1

            if not _is_mount:
                try:
                    Path(mount_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise OSError(
                        f"创建目录{mount_path}失败，请手动创建{mount_path}!"
                    ) from e

                # check nfs-common
                command_res = os.popen("dpkg -l | grep nfs-common")
                command_res = command_res.readlines()
                if not command_res:
                    raise OSError("未找到nfs-common，请通过执行以下命令安装: sudo apt install nfs-common")
                # manually mount
                try:
                    subprocess.run(mount_command, check=True, capture_output=True, text=True)
                    LOG.info("Mount finished.")
                except subprocess.CalledProcessError as e:
                    if e.returncode == 256:
                        raise OSError("挂载失败: 需要sudo权限或权限被拒绝") from e
                    elif e.returncode == 32512:
                        raise OSError(f"将{provider_uri}挂载到{mount_path}错误! 命令错误") from e
                    else:
                        raise OSError(f"挂载失败: {e.stderr}") from e
            else:
                LOG.warning(f"{_remote_uri}已挂载到{_mount_path}")


def init_from_yaml_conf(conf_path, **kwargs):
    """从yaml配置文件初始化

    :param conf_path: qlib配置文件的yml格式路径
    """

    if conf_path is None:
        config = {}
    else:
        with open(conf_path) as f:
            yaml = YAML(typ="safe", pure=True)
            config = yaml.load(f)
    config.update(kwargs)
    default_conf = config.pop("default_conf", "client")
    init(default_conf, **config)


def get_project_path(config_name="config.yaml", cur_path: Union[Path, str, None] = None) -> Path:
    """
    如果用户按照以下模式构建项目：
    - Qlib是项目路径中的子文件夹
    - qlib文件夹中有一个名为`config.yaml`的文件。

    例如：
        如果你的项目文件系统结构遵循这样的模式

            <project_path>/
              - config.yaml
              - ...一些文件夹...
                - qlib/

        这个函数将返回 <project_path>

        注意：这里不支持软链接。


    此方法通常用于：
    - 用户想使用相对配置路径而不是硬编码qlib配置路径

    Raises
    ------
    FileNotFoundError:
        If project path is not found
    """
    if cur_path is None:
        cur_path = Path(__file__).absolute().resolve()
    cur_path = Path(cur_path)
    while True:
        if (cur_path / config_name).exists():
            return cur_path
        if cur_path == cur_path.parent:
            raise FileNotFoundError("无法找到项目路径")
        cur_path = cur_path.parent


def auto_init(**kwargs):
    """
    此函数将按以下优先级自动初始化qlib：
    - 查找项目配置并初始化qlib
        - 解析过程将受配置文件的 `conf_type` 影响
    - 使用默认配置初始化qlib
    - 如果已经初始化则跳过

    :**kwargs: 可能包含以下参数
                cur_path: 查找项目路径的起始路径

    Here are two examples of the configuration

    示例 1)
    如果你想基于共享配置创建一个新的项目特定配置，你可以使用 `conf_type: ref`

    .. code-block:: yaml

        conf_type: ref
        qlib_cfg: '<shared_yaml_config_path>'    # 这可以为空，表示不从其他文件引用配置
        # 以下在 `qlib_cfg_update` 中的配置是项目特定的
        qlib_cfg_update:
            exp_manager:
                class: "MLflowExpManager"
                module_path: "qlib.workflow.expm"
                kwargs:
                    uri: "file://<your mlflow experiment path>"
                    default_exp_name: "Experiment"

    示例 2)
    如果你想创建一个简单的独立配置，你可以使用以下配置（即 `conf_type: origin`）

    .. code-block:: python

        exp_manager:
            class: "MLflowExpManager"
            module_path: "qlib.workflow.expm"
            kwargs:
                uri: "file://<your mlflow experiment path>"
                default_exp_name: "Experiment"

    """
    kwargs["skip_if_reg"] = kwargs.get("skip_if_reg", True)

    try:
        pp = get_project_path(cur_path=kwargs.pop("cur_path", None))
    except FileNotFoundError:
        init(**kwargs)
    else:
        logger = get_module_logger("Initialization")
        conf_pp = pp / "config.yaml"
        with conf_pp.open() as f:
            yaml = YAML(typ="safe", pure=True)
            conf = yaml.load(f)

        conf_type = conf.get("conf_type", "origin")
        if conf_type == "origin":
            # 配置类型与原始qlib配置相同
            init_from_yaml_conf(conf_pp, **kwargs)
        elif conf_type == "ref":
            # 这种配置类型在以下场景更方便：
            # - 有一个共享配置文件，你不想直接编辑它
            # - 共享配置可能稍后会更新，你不想复制它
            # - 你有一些自定义配置
            qlib_conf_path = conf.get("qlib_cfg", None)

            # 合并参数
            qlib_conf_update = conf.get("qlib_cfg_update", {})
            for k, v in kwargs.items():
                if k in qlib_conf_update:
                    logger.warning(f"`qlib_conf_update` 中的配置被 `kwargs` 中的键 '{k}' 覆盖")
            qlib_conf_update.update(kwargs)

            init_from_yaml_conf(qlib_conf_path, **qlib_conf_update)
        logger.info(f"Auto load project config: {conf_pp}")
