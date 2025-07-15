# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

__version__ = "0.9.6.99"
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
            raise NotImplementedError(f"This type of URI is not supported")

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
    # If the provider uri looks like this 172.23.233.89//data/csdesign'
    # It will be a nfs path. The client provider will be used
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

    For example:
        If your project file system structure follows such a pattern

            <project_path>/
              - config.yaml
              - ...some folders...
                - qlib/

        This folder will return <project_path>

        NOTE: link is not supported here.


    This method is often used when
    - user want to use a relative config path instead of hard-coding qlib config path in code

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
            raise FileNotFoundError("We can't find the project path")
        cur_path = cur_path.parent


def auto_init(**kwargs):
    """
    This function will init qlib automatically with following priority
    - Find the project configuration and init qlib
        - The parsing process will be affected by the `conf_type` of the configuration file
    - Init qlib with default config
    - Skip initialization if already initialized

    :**kwargs: it may contain following parameters
                cur_path: the start path to find the project path

    Here are two examples of the configuration

    Example 1)
    If you want to create a new project-specific config based on a shared configure, you can use  `conf_type: ref`

    .. code-block:: yaml

        conf_type: ref
        qlib_cfg: '<shared_yaml_config_path>'    # this could be null reference no config from other files
        # following configs in `qlib_cfg_update` is project=specific
        qlib_cfg_update:
            exp_manager:
                class: "MLflowExpManager"
                module_path: "qlib.workflow.expm"
                kwargs:
                    uri: "file://<your mlflow experiment path>"
                    default_exp_name: "Experiment"

    Example 2)
    If you want to create simple a standalone config, you can use following config(a.k.a. `conf_type: origin`)

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
            # The type of config is just like original qlib config
            init_from_yaml_conf(conf_pp, **kwargs)
        elif conf_type == "ref":
            # This config type will be more convenient in following scenario
            # - There is a shared configure file, and you don't want to edit it inplace.
            # - The shared configure may be updated later, and you don't want to copy it.
            # - You have some customized config.
            qlib_conf_path = conf.get("qlib_cfg", None)

            # merge the arguments
            qlib_conf_update = conf.get("qlib_cfg_update", {})
            for k, v in kwargs.items():
                if k in qlib_conf_update:
                    logger.warning(f"`qlib_conf_update` from conf_pp is override by `kwargs` on key '{k}'")
            qlib_conf_update.update(kwargs)

            init_from_yaml_conf(qlib_conf_path, **qlib_conf_update)
        logger.info(f"Auto load project config: {conf_pp}")
