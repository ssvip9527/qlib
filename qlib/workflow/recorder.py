# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from typing import Optional
import mlflow
import shutil
import pickle
import tempfile
import subprocess
import platform
from pathlib import Path
from datetime import datetime

from qlib.utils.serial import Serializable
from qlib.utils.exceptions import LoadObjectError
from qlib.utils.paral import AsyncCaller

from ..log import TimeInspector, get_module_logger
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository

logger = get_module_logger("workflow")
# mlflow limits the length of log_param to 500, but this caused errors when using qrun, so we extended the mlflow limit.
mlflow.utils.validation.MAX_PARAM_VAL_LENGTH = 1000


class Recorder:
    """
    This is the `Recorder` class for experiment recording, with API design similar to mlflow.
    (Link: https://mlflow.org/docs/latest/python_api/mlflow.html)

    The recorder's status can be SCHEDULED, RUNNING, FINISHED or FAILED.
    """

    # status type
    STATUS_S = "SCHEDULED"
    STATUS_R = "RUNNING"
    STATUS_FI = "FINISHED"
    STATUS_FA = "FAILED"

    def __init__(self, experiment_id, name):
        self.id = None
        self.name = name
        self.experiment_id = experiment_id
        self.start_time = None
        self.end_time = None
        self.status = Recorder.STATUS_S

    def __repr__(self):
        return "{name}(info={info})".format(name=self.__class__.__name__, info=self.info)

    def __str__(self):
        return str(self.info)

    def __hash__(self) -> int:
        return hash(self.info["id"])

    @property
    def info(self):
        output = dict()
        output["class"] = "Recorder"
        output["id"] = self.id
        output["name"] = self.name
        output["experiment_id"] = self.experiment_id
        output["start_time"] = self.start_time
        output["end_time"] = self.end_time
        output["status"] = self.status
        return output

    def set_recorder_name(self, rname):
        self.recorder_name = rname

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        """
        保存对象如预测文件或模型检查点到artifact URI。用户
        可以通过关键字参数(name:value)保存对象。

        请参考qlib.workflow:R.save_objects的文档

        参数
        ----------
        local_path : str
            如果提供，则将文件或目录保存到artifact URI。
        artifact_path=None : str
            存储在URI中的artifact的相对路径。
        """
        raise NotImplementedError(f"Please implement the `save_objects` method.")

    def load_object(self, name):
        """
        加载对象如预测文件或模型检查点。

        参数
        ----------
        name : str
            要加载的文件名。

        返回
        -------
        保存的对象。
        """
        raise NotImplementedError(f"Please implement the `load_object` method.")

    def start_run(self):
        """
        启动或恢复记录器。返回值可用作`with`块中的上下文管理器；
        否则必须调用end_run()来终止当前运行。(参见mlflow中的`ActiveRun`类)

        返回
        -------
        一个活动运行对象(例如mlflow.ActiveRun对象)。
        """
        raise NotImplementedError(f"Please implement the `start_run` method.")

    def end_run(self):
        """
        结束一个活动的记录器。
        """
        raise NotImplementedError(f"Please implement the `end_run` method.")

    def log_params(self, **kwargs):
        """
        为当前运行记录一批参数。

        参数
        ----------
        关键字参数
            要记录为参数的键值对。
        """
        raise NotImplementedError(f"Please implement the `log_params` method.")

    def log_metrics(self, step=None, **kwargs):
        """
        为当前运行记录多个指标。

        参数
        ----------
        关键字参数
            要记录为指标的键值对。
        """
        raise NotImplementedError(f"Please implement the `log_metrics` method.")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        将本地文件或目录记录为当前活动运行的artifact。

        参数
        ----------
        local_path : str
            要写入的文件路径。
        artifact_path : Optional[str]
            如果提供，则写入到``artifact_uri``中的目录。
        """
        raise NotImplementedError(f"Please implement the `log_metrics` method.")

    def set_tags(self, **kwargs):
        """
        为当前运行记录一批标签。

        参数
        ----------
        关键字参数
            要记录为标签的键值对。
        """
        raise NotImplementedError(f"Please implement the `set_tags` method.")

    def delete_tags(self, *keys):
        """
        从运行中删除一些标签。

        参数
        ----------
        keys : 键的字符串序列
            要删除的所有标签名称。
        """
        raise NotImplementedError(f"Please implement the `delete_tags` method.")

    def list_artifacts(self, artifact_path: str = None):
        """
        列出记录器的所有artifacts。

        参数
        ----------
        artifact_path : str
            artifact存储在URI中的相对路径。

        返回
        -------
            存储的artifacts信息列表(名称、路径等)。
        """
        raise NotImplementedError(f"Please implement the `list_artifacts` method.")

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        """
        从运行中下载artifact文件或目录到本地目录(如果适用)，
        并返回其本地路径。

        参数
        ----------
        path : str
            目标artifact的相对源路径。
        dst_path : Optional[str]
            本地文件系统目标目录的绝对路径，用于
            下载指定的artifacts。该目录必须已存在。
            如果未指定，artifacts将被下载到本地文件系统上
            一个唯一命名的新目录中。

        返回
        -------
        str
            目标artifact的本地路径。
        """
        raise NotImplementedError(f"Please implement the `list_artifacts` method.")

    def list_metrics(self):
        """
        列出记录器的所有指标。

        返回
        -------
            存储的指标字典。
        """
        raise NotImplementedError(f"Please implement the `list_metrics` method.")

    def list_params(self):
        """
        列出记录器的所有参数。

        返回
        -------
            存储的参数字典。
        """
        raise NotImplementedError(f"Please implement the `list_params` method.")

    def list_tags(self):
        """
        列出记录器的所有标签。

        返回
        -------
            存储的标签字典。
        """
        raise NotImplementedError(f"Please implement the `list_tags` method.")


class MLflowRecorder(Recorder):
    """
    使用mlflow实现一个Recorder。

    由于mlflow只能从文件或目录记录artifact，我们决定使用
    文件管理器来帮助维护项目中的对象。

    我们不是直接使用mlflow，而是使用另一个包装mlflow的接口来记录实验。
    虽然需要额外的工作，但由于以下原因它为用户带来了好处：
    - 可以更方便地更改实验记录后端，而无需更改上层代码
    - 我们可以提供更多便利来自动执行一些额外操作并使接口更简单。例如：
        - 自动记录未提交的代码
        - 自动记录部分环境变量
        - 用户只需创建不同的Recorder即可控制多个不同的运行(在mlflow中，您总是需要频繁切换artifact_uri并传入运行id)
    """

    def __init__(self, experiment_id, uri, name=None, mlflow_run=None):
        super(MLflowRecorder, self).__init__(experiment_id, name)
        self._uri = uri
        self._artifact_uri = None
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        # construct from mlflow run
        if mlflow_run is not None:
            assert isinstance(mlflow_run, mlflow.entities.run.Run), "Please input with a MLflow Run object."
            self.name = mlflow_run.data.tags["mlflow.runName"]
            self.id = mlflow_run.info.run_id
            self.status = mlflow_run.info.status
            self.start_time = (
                datetime.fromtimestamp(float(mlflow_run.info.start_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                if mlflow_run.info.start_time is not None
                else None
            )
            self.end_time = (
                datetime.fromtimestamp(float(mlflow_run.info.end_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                if mlflow_run.info.end_time is not None
                else None
            )
            self._artifact_uri = mlflow_run.info.artifact_uri
        self.async_log = None

    def __repr__(self):
        name = self.__class__.__name__
        space_length = len(name) + 1
        return "{name}(info={info},\n{space}uri={uri},\n{space}artifact_uri={artifact_uri},\n{space}client={client})".format(
            name=name,
            space=" " * space_length,
            info=self.info,
            uri=self.uri,
            artifact_uri=self.artifact_uri,
            client=self.client,
        )

    def __hash__(self) -> int:
        return hash(self.info["id"])

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MLflowRecorder):
            return self.info["id"] == o.info["id"]
        return False

    @property
    def uri(self):
        return self._uri

    @property
    def artifact_uri(self):
        return self._artifact_uri

    def get_local_dir(self):
        """
        此函数将返回此记录器的目录路径。
        """
        if self.artifact_uri is not None:
            if platform.system() == "Windows":
                local_dir_path = Path(self.artifact_uri.lstrip("file:").lstrip("/")).parent
            else:
                local_dir_path = Path(self.artifact_uri.lstrip("file:")).parent
            local_dir_path = str(local_dir_path.resolve())
            if os.path.isdir(local_dir_path):
                return local_dir_path
            else:
                raise RuntimeError("This recorder is not saved in the local file system.")

        else:
            raise ValueError(
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            )

    def start_run(self):
        # set the tracking uri
        mlflow.set_tracking_uri(self.uri)
        # start the run
        run = mlflow.start_run(self.id, self.experiment_id, self.name)
        # save the run id and artifact_uri
        self.id = run.info.run_id
        self._artifact_uri = run.info.artifact_uri
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status = Recorder.STATUS_R
        logger.info(f"Recorder {self.id} starts running under Experiment {self.experiment_id} ...")

        # NOTE: making logging async.
        # - This may cause delay when uploading results
        # - The logging time may not be accurate
        self.async_log = AsyncCaller()

        # TODO: currently, this is only supported in MLflowRecorder.
        # Maybe we can make this feature more general.
        self._log_uncommitted_code()

        self.log_params(**{"cmd-sys.argv": " ".join(sys.argv)})  # log the command to produce current experiment
        self.log_params(
            **{k: v for k, v in os.environ.items() if k.startswith("_QLIB_")}
        )  # Log necessary environment variables
        return run

    def _log_uncommitted_code(self):
        """
        Mlflow只记录当前仓库的提交ID。但通常用户会有很多未提交的更改。
        因此这个方法尝试自动记录所有这些更改。
        """
        # TODO: the sub-directories maybe git repos.
        # So it will be better if we can walk the sub-directories and log the uncommitted changes.
        for cmd, fname in [
            ("git diff", "code_diff.txt"),
            ("git status", "code_status.txt"),
            ("git diff --cached", "code_cached.txt"),
        ]:
            try:
                out = subprocess.check_output(cmd, shell=True)
                self.client.log_text(self.id, out.decode(), fname)  # this behaves same as above
            except subprocess.CalledProcessError:
                logger.info(f"Fail to log the uncommitted code of $CWD({os.getcwd()}) when run {cmd}.")

    def end_run(self, status: str = Recorder.STATUS_S):
        assert status in [
            Recorder.STATUS_S,
            Recorder.STATUS_R,
            Recorder.STATUS_FI,
            Recorder.STATUS_FA,
        ], f"The status type {status} is not supported."
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.status != Recorder.STATUS_S:
            self.status = status
        if self.async_log is not None:
            # Waiting Queue should go before mlflow.end_run. Otherwise mlflow will raise error
            with TimeInspector.logt("waiting `async_log`"):
                self.async_log.wait()
        self.async_log = None
        mlflow.end_run(status)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."
        if local_path is not None:
            path = Path(local_path)
            if path.is_dir():
                self.client.log_artifacts(self.id, local_path, artifact_path)
            else:
                self.client.log_artifact(self.id, local_path, artifact_path)
        else:
            temp_dir = Path(tempfile.mkdtemp()).resolve()
            for name, data in kwargs.items():
                path = temp_dir / name
                Serializable.general_dump(data, path)
                self.client.log_artifact(self.id, temp_dir / name, artifact_path)
            shutil.rmtree(temp_dir)

    def load_object(self, name, unpickler=pickle.Unpickler):
        """
        从mlflow加载对象如预测文件或模型检查点。

        参数:
            name (str): 对象名称

            unpickler: 支持使用自定义unpickler

        抛出:
            LoadObjectError: 如果加载对象时出现异常

        返回:
            object: mlflow中保存的对象。
        """
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."

        path = None
        try:
            path = self.client.download_artifacts(self.id, name)
            with Path(path).open("rb") as f:
                data = unpickler(f).load()
            return data
        except Exception as e:
            raise LoadObjectError(str(e)) from e
        finally:
            ar = self.client._tracking_client._get_artifact_repo(self.id)
            if isinstance(ar, AzureBlobArtifactRepository) and path is not None:
                # for saving disk space
                # For safety, only remove redundant file for specific ArtifactRepository
                shutil.rmtree(Path(path).absolute().parent)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_params(self, **kwargs):
        for name, data in kwargs.items():
            self.client.log_param(self.id, name, data)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metrics(self, step=None, **kwargs):
        for name, data in kwargs.items():
            self.client.log_metric(self.id, name, data, step=step)

    def log_artifact(self, local_path, artifact_path: Optional[str] = None):
        self.client.log_artifact(self.id, local_path=local_path, artifact_path=artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def set_tags(self, **kwargs):
        for name, data in kwargs.items():
            self.client.set_tag(self.id, name, data)

    def delete_tags(self, *keys):
        for key in keys:
            self.client.delete_tag(self.id, key)

    def get_artifact_uri(self):
        if self.artifact_uri is not None:
            return self.artifact_uri
        else:
            raise ValueError(
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            )

    def list_artifacts(self, artifact_path=None):
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."
        artifacts = self.client.list_artifacts(self.id, artifact_path)
        return [art.path for art in artifacts]

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        return self.client.download_artifacts(self.id, path, dst_path)

    def list_metrics(self):
        run = self.client.get_run(self.id)
        return run.data.metrics

    def list_params(self):
        run = self.client.get_run(self.id)
        return run.data.params

    def list_tags(self):
        run = self.client.get_run(self.id)
        return run.data.tags
