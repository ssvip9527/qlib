# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Union
from qlib.typehint import Literal
import mlflow
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from .recorder import Recorder, MLflowRecorder
from ..log import get_module_logger

logger = get_module_logger("workflow")


class Experiment:
    """
    这是用于运行每个实验的`Experiment`类。API设计类似于mlflow。
    (链接: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.active_recorder = None  # only one recorder can run each time
        self._default_rec_name = "abstract_recorder"

    def __repr__(self):
        return "{name}(id={id}, info={info})".format(name=self.__class__.__name__, id=self.id, info=self.info)

    def __str__(self):
        return str(self.info)

    @property
    def info(self):
        recorders = self.list_recorders()
        output = dict()
        output["class"] = "Experiment"
        output["id"] = self.id
        output["name"] = self.name
        output["active_recorder"] = self.active_recorder.id if self.active_recorder is not None else None
        output["recorders"] = list(recorders.keys())
        return output

    def start(self, *, recorder_id=None, recorder_name=None, resume=False):
        """
        开始实验并设置为活动状态。此方法还将启动一个新的记录器。

        参数
        ----------
        recorder_id : str
            要创建的记录器ID
        recorder_name : str
            要创建的记录器名称
        resume : bool
            是否恢复第一个记录器

        返回
        -------
        一个活动的记录器。
        """
        raise NotImplementedError(f"Please implement the `start` method.")

    def end(self, recorder_status=Recorder.STATUS_S):
        """
        结束实验。

        参数
        ----------
        recorder_status : str
            结束时记录器要设置的状态(SCHEDULED, RUNNING, FINISHED, FAILED)。
        """
        raise NotImplementedError(f"Please implement the `end` method.")

    def create_recorder(self, recorder_name=None):
        """
        为每个实验创建记录器

        Parameters
        ----------
        recorder_name : str
            要创建记录器的名称

        Returns
        -------
        记录器对象
        """
        raise NotImplementedError(f"Please implement the `create_recorder` method.")

    def search_records(self, **kwargs):
        """
        获取符合实验搜索条件的记录DataFrame
        输入为用户想要应用的搜索条件

        Returns
        -------
        包含记录的pandas.DataFrame，其中每个指标、参数和标签
        都被展开到名为metrics.*、params.*和tags.*的列中
        对于没有特定指标、参数或标签的记录，它们的值将分别为(NumPy)Nan、None或None
        """
        raise NotImplementedError(f"Please implement the `search_records` method.")

    def delete_recorder(self, recorder_id):
        """
        为每个实验创建记录器。

        参数
        ----------
        recorder_id : str
            要删除的记录器ID。
        """
        raise NotImplementedError(f"Please implement the `delete_recorder` method.")

    def get_recorder(self, recorder_id=None, recorder_name=None, create: bool = True, start: bool = False) -> Recorder:
        """
        为用户检索记录器。当用户指定记录器ID和名称时，该方法会尝试返回特定的记录器。
        当用户未提供记录器ID或名称时，该方法会尝试返回当前活动记录器。
        `create`参数决定如果记录器尚未创建，该方法是否会根据用户规范自动创建新记录器。

        * 如果`create`为True:

            * 如果`活动记录器`存在:

                * 未指定ID或名称，返回活动记录器。
                * 如果指定了ID或名称，返回指定的记录器。如果未找到对应实验，则使用给定的ID或名称创建新记录器。如果`start`设为True，则将记录器设为活动状态。

            * 如果`活动记录器`不存在:

                * 未指定ID或名称，创建新记录器。
                * 如果指定了ID或名称，返回指定的实验。如果未找到对应实验，则使用给定的ID或名称创建新记录器。如果`start`设为True，则将记录器设为活动状态。

        * 如果`create`为False:

            * 如果`活动记录器`存在:

                * 未指定ID或名称，返回活动记录器。
                * 如果指定了ID或名称，返回指定的记录器。如果未找到对应实验，则抛出错误。

            * 如果`活动记录器`不存在:

                * 未指定ID或名称，抛出错误。
                * 如果指定了ID或名称，返回指定的记录器。如果未找到对应实验，则抛出错误。

        参数
        ----------
        recorder_id : str
            要删除的记录器ID。
        recorder_name : str
            要删除的记录器名称。
        create : boolean
            如果记录器尚未创建，则创建它。
        start : boolean
            如果创建了新记录器，则启动它。

        返回
        -------
        记录器对象。
        """
        # special case of getting the recorder
        if recorder_id is None and recorder_name is None:
            if self.active_recorder is not None:
                return self.active_recorder
            recorder_name = self._default_rec_name
        if create:
            recorder, is_new = self._get_or_create_rec(recorder_id=recorder_id, recorder_name=recorder_name)
        else:
            recorder, is_new = (
                self._get_recorder(recorder_id=recorder_id, recorder_name=recorder_name),
                False,
            )
        if is_new and start:
            self.active_recorder = recorder
            # start the recorder
            self.active_recorder.start_run()
        return recorder

    def _get_or_create_rec(self, recorder_id=None, recorder_name=None) -> (object, bool):
        """
        获取或创建记录器的方法。首先尝试获取有效的记录器，如果发生异常，
        则根据给定的ID和名称自动创建新记录器。
        """
        try:
            if recorder_id is None and recorder_name is None:
                recorder_name = self._default_rec_name
            return (
                self._get_recorder(recorder_id=recorder_id, recorder_name=recorder_name),
                False,
            )
        except ValueError:
            if recorder_name is None:
                recorder_name = self._default_rec_name
            logger.info(f"No valid recorder found. Create a new recorder with name {recorder_name}.")
            return self.create_recorder(recorder_name), True

    def _get_recorder(self, recorder_id=None, recorder_name=None):
        """
        通过名称或ID获取特定记录器。如果不存在，则抛出ValueError

        参数
        ----------
        recorder_id :
            记录器ID
        recorder_name :
            记录器名称

        返回
        -------
        Recorder:
            搜索到的记录器

        Raises
        ------
        ValueError
        """
        raise NotImplementedError(f"Please implement the `_get_recorder` method")

    RT_D = "dict"  # return type dict
    RT_L = "list"  # return type list

    def list_recorders(
        self, rtype: Literal["dict", "list"] = RT_D, **flt_kwargs
    ) -> Union[List[Recorder], Dict[str, Recorder]]:
        """
        列出本实验的所有现有记录器。调用此方法前请先获取实验实例。
        如果想使用`R.list_recorders()`方法，请参考`QlibRecorder`中的相关API文档。

        flt_kwargs : dict
            按条件过滤记录器
            例如：list_recorders(status=Recorder.STATUS_FI)

        返回
        -------
        返回类型取决于`rtype`
            如果`rtype` == "dict":
                存储的记录器信息的字典(id -> recorder)。
            如果`rtype` == "list":
                记录器列表。
        """
        raise NotImplementedError(f"Please implement the `list_recorders` method.")


class MLflowExperiment(Experiment):
    """
    使用mlflow实现Experiment。
    """

    def __init__(self, id, name, uri):
        super(MLflowExperiment, self).__init__(id, name)
        self._uri = uri
        self._default_rec_name = "mlflow_recorder"
        self._client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)

    def __repr__(self):
        return "{name}(id={id}, info={info})".format(name=self.__class__.__name__, id=self.id, info=self.info)

    def start(self, *, recorder_id=None, recorder_name=None, resume=False):
        logger.info(f"Experiment {self.id} starts running ...")
        # Get or create recorder
        if recorder_name is None:
            recorder_name = self._default_rec_name
        # resume the recorder
        if resume:
            recorder, _ = self._get_or_create_rec(recorder_id=recorder_id, recorder_name=recorder_name)
        # create a new recorder
        else:
            recorder = self.create_recorder(recorder_name)
        # Set up active recorder
        self.active_recorder = recorder
        # Start the recorder
        self.active_recorder.start_run()

        return self.active_recorder

    def end(self, recorder_status=Recorder.STATUS_S):
        if self.active_recorder is not None:
            self.active_recorder.end_run(recorder_status)
            self.active_recorder = None

    def create_recorder(self, recorder_name=None):
        if recorder_name is None:
            recorder_name = self._default_rec_name
        recorder = MLflowRecorder(self.id, self._uri, recorder_name)

        return recorder

    def _get_recorder(self, recorder_id=None, recorder_name=None):
        """
        Method for getting or creating a recorder. It will try to first get a valid recorder, if exception occurs, it will
        raise errors.

        Quoting docs of search_runs from MLflow
        > The default ordering is to sort by start_time DESC, then run_id.
        """
        assert (
            recorder_id is not None or recorder_name is not None
        ), "Please input at least one of recorder id or name before retrieving recorder."
        if recorder_id is not None:
            try:
                run = self._client.get_run(recorder_id)
                recorder = MLflowRecorder(self.id, self._uri, mlflow_run=run)
                return recorder
            except MlflowException as mlflow_exp:
                raise ValueError("未找到有效的记录器，请确保输入的记录器ID正确。") from mlflow_exp
        elif recorder_name is not None:
            logger.warning(f"请确保记录器名称{recorder_name}是唯一的，如果存在多个匹配给定名称的记录器，我们只会返回最新的一个。")
            recorders = self.list_recorders()
            for rid in recorders:
                if recorders[rid].name == recorder_name:
                    return recorders[rid]
            raise ValueError("未找到有效的记录器，请确保输入的记录器名称正确。")

    def search_records(self, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")

        return self._client.search_runs([self.id], filter_string, run_view_type, max_results, order_by)

    def delete_recorder(self, recorder_id=None, recorder_name=None):
        assert (
            recorder_id is not None or recorder_name is not None
        ), "删除前请输入有效的记录器ID或名称。"
        try:
            if recorder_id is not None:
                self._client.delete_run(recorder_id)
            else:
                recorder = self._get_recorder(recorder_name=recorder_name)
                self._client.delete_run(recorder.id)
        except MlflowException as e:
            raise ValueError(
                f"错误: {e}. 删除记录器时出现问题。请检查记录器的`name/id`是否正确。"
            ) from e

    UNLIMITED = 50000  # FIXME: Mlflow can only list 50000 records at most!!!!!!!

    def list_recorders(
        self,
        rtype: Literal["dict", "list"] = Experiment.RT_D,
        max_results: int = UNLIMITED,
        status: Union[str, None] = None,
        filter_string: str = "",
    ):
        """
        引用search_runs文档
        > 默认排序是按start_time降序，然后按run_id。

        参数
        ----------
        max_results : int
            结果数量限制
        status : str
            基于状态过滤结果的标准。
            `None`表示不过滤。
        filter_string : str
            mlflow支持的过滤字符串，如'params."my_param"="a" and tags."my_tag"="b"'，使用此参数有助于减少运行数量。
        """
        runs = self._client.search_runs(
            self.id, run_view_type=ViewType.ACTIVE_ONLY, max_results=max_results, filter_string=filter_string
        )
        rids = []
        recorders = []
        for i, n in enumerate(runs):
            recorder = MLflowRecorder(self.id, self._uri, mlflow_run=n)
            if status is None or recorder.status == status:
                rids.append(n.info.run_id)
                recorders.append(recorder)

        if rtype == Experiment.RT_D:
            return dict(zip(rids, recorders))
        elif rtype == Experiment.RT_L:
            return recorders
        else:
            raise NotImplementedError(f"This type of input is not supported")
