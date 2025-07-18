# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from urllib.parse import urlparse
import mlflow
from filelock import FileLock
from mlflow.exceptions import MlflowException, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.entities import ViewType
import os
from typing import Optional, Text
from pathlib import Path

from .exp import MLflowExperiment, Experiment
from ..config import C
from .recorder import Recorder
from ..log import get_module_logger
from ..utils.exceptions import ExpAlreadyExistError


logger = get_module_logger("workflow")


class ExpManager:
    """
    这是用于管理实验的`ExpManager`类。API设计类似于mlflow。
    (链接: https://mlflow.org/docs/latest/python_api/mlflow.html)

    `ExpManager`预期是一个单例(同时，我们可以有多个具有不同uri的`Experiment`。用户可以从不同的uri获取不同的实验，然后比较它们的记录)。全局配置(即`C`)也是一个单例。

    因此我们尝试将它们对齐。它们共享同一个变量，称为**默认uri**。有关变量共享的详细信息，请参阅`ExpManager.default_uri`。

    当用户开始一个实验时，用户可能希望将uri设置为特定的uri(在此期间它将覆盖**默认uri**)，然后取消设置**特定uri**并回退到**默认uri**。`ExpManager._active_exp_uri`就是那个**特定uri**。
    """

    active_experiment: Optional[Experiment]

    def __init__(self, uri: Text, default_exp_name: Optional[Text]):
        self.default_uri = uri
        self._active_exp_uri = None  # No active experiments. So it is set to None
        self._default_exp_name = default_exp_name
        self.active_experiment = None  # only one experiment can be active each time
        logger.debug(f"experiment manager uri is at {self.uri}")

    def __repr__(self):
        return "{name}(uri={uri})".format(name=self.__class__.__name__, uri=self.uri)

    def start_exp(
        self,
        *,
        experiment_id: Optional[Text] = None,
        experiment_name: Optional[Text] = None,
        recorder_id: Optional[Text] = None,
        recorder_name: Optional[Text] = None,
        uri: Optional[Text] = None,
        resume: bool = False,
        **kwargs,
    ) -> Experiment:
        """
        启动一个实验。该方法首先获取或创建一个实验，然后将其设置为活动状态。

        `_active_exp_uri`的维护包含在start_exp中，剩余实现应包含在子类的_end_exp中

        参数
        ----------
        experiment_id : str
            活动实验的ID
        experiment_name : str
            活动实验的名称
        recorder_id : str
            要启动的记录器ID
        recorder_name : str
            要启动的记录器名称
        uri : str
            当前跟踪URI
        resume : boolean
            是否恢复实验和记录器

        返回
        -------
        一个活动实验对象
        """
        self._active_exp_uri = uri
        # The subclass may set the underlying uri back.
        # So setting `_active_exp_uri` come before `_start_exp`
        return self._start_exp(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            recorder_id=recorder_id,
            recorder_name=recorder_name,
            resume=resume,
            **kwargs,
        )

    def _start_exp(self, *args, **kwargs) -> Experiment:
        """请参考`start_exp`方法的文档"""
        raise NotImplementedError(f"Please implement the `start_exp` method.")

    def end_exp(self, recorder_status: Text = Recorder.STATUS_S, **kwargs):
        """
        结束一个活动实验

        `_active_exp_uri`的维护包含在end_exp中，剩余实现应包含在子类的_end_exp中

        参数
        ----------
        experiment_name : str
            活动实验的名称
        recorder_status : str
            实验活动记录器的状态
        """
        self._active_exp_uri = None
        # The subclass may set the underlying uri back.
        # So setting `_active_exp_uri` come before `_end_exp`
        self._end_exp(recorder_status=recorder_status, **kwargs)

    def _end_exp(self, recorder_status: Text = Recorder.STATUS_S, **kwargs):
        raise NotImplementedError(f"请实现`end_exp`方法")

    def create_exp(self, experiment_name: Optional[Text] = None):
        """
        创建一个实验

        参数
        ----------
        experiment_name : str
            实验名称，必须唯一

        返回
        -------
        一个实验对象

        Raise
        -----
        ExpAlreadyExistError
            当实验已存在时抛出
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def search_records(self, experiment_ids=None, **kwargs):
        """
        获取符合实验搜索条件的记录DataFrame
        输入为用户想要应用的搜索条件

        返回
        -------
        一个pandas.DataFrame记录，其中每个指标、参数和标签
        分别展开到名为metrics.*、params.*和tags.*的列中
        对于没有特定指标、参数或标签的记录，它们的值将分别为(NumPy)Nan、None或None
        """
        raise NotImplementedError(f"Please implement the `search_records` method.")

    def get_exp(self, *, experiment_id=None, experiment_name=None, create: bool = True, start: bool = False):
        """
        检索一个实验。该方法包括获取活动实验，以及获取或创建特定实验

        当用户指定实验ID和名称时，方法将尝试返回特定实验
        当用户未提供记录器ID或名称时，方法将尝试返回当前活动实验
        `create`参数决定如果实验尚未创建，方法是否根据用户规范自动创建新实验

        * 如果`create`为True:

            * 如果`活动实验`存在:

                * 未指定ID或名称，返回活动实验
                * 如果指定了ID或名称，返回指定实验。如果未找到，则使用给定ID或名称创建新实验。如果`start`设为True，实验将被设置为活动状态

            * 如果`活动实验`不存在:

                * 未指定ID或名称，创建默认实验
                * 如果指定了ID或名称，返回指定实验。如果未找到，则使用给定ID或名称创建新实验。如果`start`设为True，实验将被设置为活动状态

        * 如果`create`为False:

            * 如果`活动实验`存在:

                * 未指定ID或名称，返回活动实验
                * 如果指定了ID或名称，返回指定实验。如果未找到，抛出错误

            * 如果`活动实验`不存在:

                * 未指定ID或名称。如果默认实验存在则返回，否则抛出错误
                * 如果指定了ID或名称，返回指定实验。如果未找到，抛出错误

        参数
        ----------
        experiment_id : str
            要返回的实验ID
        experiment_name : str
            要返回的实验名称
        create : boolean
            如果实验尚未创建，是否创建它
        start : boolean
            如果创建了新实验，是否启动它

        返回
        -------
        一个实验对象
        """
        # special case of getting experiment
        if experiment_id is None and experiment_name is None:
            if self.active_experiment is not None:
                return self.active_experiment
            # User don't want get active code now.
            experiment_name = self._default_exp_name

        if create:
            exp, _ = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        else:
            exp = self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        if self.active_experiment is None and start:
            self.active_experiment = exp
            # start the recorder
            self.active_experiment.start()
        return exp

    def _get_or_create_exp(self, experiment_id=None, experiment_name=None) -> (object, bool):
        """
        获取或创建实验的方法。首先尝试获取有效实验，如果发生异常，
        则根据给定的ID和名称自动创建新实验。
        """
        try:
            return (
                self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name),
                False,
            )
        except ValueError:
            if experiment_name is None:
                experiment_name = self._default_exp_name
            logger.warning(f"No valid experiment found. Create a new experiment with name {experiment_name}.")

            # NOTE: mlflow doesn't consider the lock for recording multiple runs
            # So we supported it in the interface wrapper
            pr = urlparse(self.uri)
            if pr.scheme == "file":
                with FileLock(Path(os.path.join(pr.netloc, pr.path.lstrip("/"), "filelock"))):  # pylint: disable=E0110
                    return self.create_exp(experiment_name), True
            # NOTE: for other schemes like http, we double check to avoid create exp conflicts
            try:
                return self.create_exp(experiment_name), True
            except ExpAlreadyExistError:
                return (
                    self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name),
                    False,
                )

    def _get_exp(self, experiment_id=None, experiment_name=None) -> Experiment:
        """
        通过名称或ID获取特定实验。如果不存在则抛出ValueError

        参数
        ----------
        experiment_id :
            实验ID
        experiment_name :
            实验名称

        返回
        -------
        Experiment:
            搜索到的实验

        Raises
        ------
        ValueError
            当实验不存在时抛出
        """
        raise NotImplementedError(f"Please implement the `_get_exp` method")

    def delete_exp(self, experiment_id=None, experiment_name=None):
        """
        删除一个实验

        参数
        ----------
        experiment_id  : str
            实验ID
        experiment_name  : str
            实验名称
        """
        raise NotImplementedError(f"Please implement the `delete_exp` method.")

    @property
    def default_uri(self):
        """
        从qlib.config.C获取默认跟踪URI
        """
        if "kwargs" not in C.exp_manager or "uri" not in C.exp_manager["kwargs"]:
            raise ValueError("The default URI is not set in qlib.config.C")
        return C.exp_manager["kwargs"]["uri"]

    @default_uri.setter
    def default_uri(self, value):
        C.exp_manager.setdefault("kwargs", {})["uri"] = value

    @property
    def uri(self):
        """
        获取默认跟踪URI或当前URI

        返回
        -------
        跟踪URI字符串
        """
        return self._active_exp_uri or self.default_uri

    def list_experiments(self):
        """
        列出所有现有实验

        返回
        -------
        存储的实验信息字典(name -> experiment)
        """
        raise NotImplementedError(f"Please implement the `list_experiments` method.")


class MLflowExpManager(ExpManager):
    """
    使用mlflow实现ExpManager
    """

    @property
    def client(self):
        # 请参考`tests/dependency_tests/test_mlflow.py::MLflowTest::test_creating_client`
        # 该测试确保创建新客户端的速度
        return mlflow.tracking.MlflowClient(tracking_uri=self.uri)

    def _start_exp(
        self,
        *,
        experiment_id: Optional[Text] = None,
        experiment_name: Optional[Text] = None,
        recorder_id: Optional[Text] = None,
        recorder_name: Optional[Text] = None,
        resume: bool = False,
    ):
        # Create experiment
        if experiment_name is None:
            experiment_name = self._default_exp_name
        experiment, _ = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        # Set up active experiment
        self.active_experiment = experiment
        # Start the experiment
        self.active_experiment.start(recorder_id=recorder_id, recorder_name=recorder_name, resume=resume)

        return self.active_experiment

    def _end_exp(self, recorder_status: Text = Recorder.STATUS_S):
        if self.active_experiment is not None:
            self.active_experiment.end(recorder_status)
            self.active_experiment = None

    def create_exp(self, experiment_name: Optional[Text] = None):
        assert experiment_name is not None
        # init experiment
        try:
            experiment_id = self.client.create_experiment(experiment_name)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise ExpAlreadyExistError() from e
            raise e

        return MLflowExperiment(experiment_id, experiment_name, self.uri)

    def _get_exp(self, experiment_id=None, experiment_name=None):
        """
        获取或创建实验的方法。首先尝试获取有效实验，如果发生异常则抛出错误。
        """
        assert (
            experiment_id is not None or experiment_name is not None
        ), "Please input at least one of experiment/recorder id or name before retrieving experiment/recorder."
        if experiment_id is not None:
            try:
                # NOTE: the mlflow's experiment_id must be str type...
                # https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.get_experiment
                exp = self.client.get_experiment(experiment_id)
                if exp.lifecycle_stage.upper() == "DELETED":
                    raise MlflowException("No valid experiment has been found.")
                experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
                return experiment
            except MlflowException as e:
                raise ValueError(
                    "No valid experiment has been found, please make sure the input experiment id is correct."
                ) from e
        elif experiment_name is not None:
            try:
                exp = self.client.get_experiment_by_name(experiment_name)
                if exp is None or exp.lifecycle_stage.upper() == "DELETED":
                    raise MlflowException("No valid experiment has been found.")
                experiment = MLflowExperiment(exp.experiment_id, experiment_name, self.uri)
                return experiment
            except MlflowException as e:
                raise ValueError(
                    "No valid experiment has been found, please make sure the input experiment name is correct."
                ) from e

    def search_records(self, experiment_ids=None, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")
        return self.client.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)

    def delete_exp(self, experiment_id=None, experiment_name=None):
        assert (
            experiment_id is not None or experiment_name is not None
        ), "Please input a valid experiment id or name before deleting."
        try:
            if experiment_id is not None:
                self.client.delete_experiment(experiment_id)
            else:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise MlflowException("No valid experiment has been found.")
                self.client.delete_experiment(experiment.experiment_id)
        except MlflowException as e:
            raise ValueError(
                f"Error: {e}. Something went wrong when deleting experiment. Please check if the name/id of the experiment is correct."
            ) from e

    def list_experiments(self):
        # retrieve all the existing experiments
        mlflow_version = int(mlflow.__version__.split(".", maxsplit=1)[0])
        if mlflow_version >= 2:
            exps = self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        else:
            exps = self.client.list_experiments(view_type=ViewType.ACTIVE_ONLY)  # pylint: disable=E1101
        experiments = dict()
        for exp in exps:
            experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
            experiments[exp.name] = experiment
        return experiments
