# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
设计动机(相比直接使用mlflow):
- 比mlflow原生设计更好
    - 我们拥有包含丰富方法的记录对象(更直观)，而不是像mlflow中每次都要使用run_id
        - 因此记录器的接口如log、start等会更加直观
- 提供比mlflow原生更丰富和定制化的功能
    - 在运行开始时记录代码差异
    - 直接使用log_object和load_object处理Python对象，而不是log_artifact和download_artifact
- (较弱)支持多种后端

诚然，设计总会带来额外负担。例如：
- 必须先创建实验才能获取记录器(在MLflow中，实验更像是标签，您通常可以在许多接口中直接使用run_id而无需先定义实验)
"""

from contextlib import contextmanager
from typing import Text, Optional, Any, Dict
from .expm import ExpManager
from .exp import Experiment
from .recorder import Recorder
from ..utils import Wrapper
from ..utils.exceptions import RecorderInitializationError


class QlibRecorder:
    """
    用于管理实验的全局系统。
    """

    def __init__(self, exp_manager: ExpManager):
        self.exp_manager: ExpManager = exp_manager

    def __repr__(self):
        return "{name}(manager={manager})".format(name=self.__class__.__name__, manager=self.exp_manager)

    @contextmanager
    def start(
        self,
        *,
        experiment_id: Optional[Text] = None,
        experiment_name: Optional[Text] = None,
        recorder_id: Optional[Text] = None,
        recorder_name: Optional[Text] = None,
        uri: Optional[Text] = None,
        resume: bool = False,
    ):
        """
        启动实验的方法。此方法只能在Python的`with`语句中调用。示例代码如下：

        .. code-block:: Python

            # 启动新实验和记录器
            with R.start(experiment_name='test', recorder_name='recorder_1'):
                model.fit(dataset)
                R.log...
                ... # 其他操作

            # 恢复之前的实验和记录器
            with R.start(experiment_name='test', recorder_name='recorder_1', resume=True): # 如果用户想恢复记录器，必须指定完全相同的实验和记录器名称
                ... # 其他操作


        参数
        ----------
        experiment_id : str
            要启动的实验ID
        experiment_name : str
            要启动的实验名称
        recorder_id : str
            实验下要启动的记录器ID
        recorder_name : str
            实验下要启动的记录器名称
        uri : str
            实验的跟踪URI，所有artifacts/metrics等将存储在此处
            默认URI设置在qlib.config中。注意此uri参数不会更改配置文件中的设置
            因此下次在同一实验中调用此函数时，用户必须指定相同的值，否则可能出现URI不一致
        resume : bool
            是否恢复指定名称的记录器
        """
        run = self.start_exp(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            recorder_id=recorder_id,
            recorder_name=recorder_name,
            uri=uri,
            resume=resume,
        )
        try:
            yield run
        except Exception as e:
            self.end_exp(Recorder.STATUS_FA)  # end the experiment if something went wrong
            raise e
        self.end_exp(Recorder.STATUS_FI)

    def start_exp(
        self,
        *,
        experiment_id=None,
        experiment_name=None,
        recorder_id=None,
        recorder_name=None,
        uri=None,
        resume=False,
    ):
        """
        Lower level method for starting an experiment. When use this method, one should end the experiment manually
        and the status of the recorder may not be handled properly. Here is the example code:

        .. code-block:: Python

            R.start_exp(experiment_name='test', recorder_name='recorder_1')
            ... # further operations
            R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)


        Parameters
        ----------
        experiment_id : str
            id of the experiment one wants to start.
        experiment_name : str
            the name of the experiment to be started
        recorder_id : str
            id of the recorder under the experiment one wants to start.
        recorder_name : str
            name of the recorder under the experiment one wants to start.
        uri : str
            the tracking uri of the experiment, where all the artifacts/metrics etc. will be stored.
            The default uri are set in the qlib.config.
        resume : bool
            whether to resume the specific recorder with given name under the given experiment.

        Returns
        -------
        An experiment instance being started.
        """
        return self.exp_manager.start_exp(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            recorder_id=recorder_id,
            recorder_name=recorder_name,
            uri=uri,
            resume=resume,
        )

    def end_exp(self, recorder_status=Recorder.STATUS_FI):
        """
        Method for ending an experiment manually. It will end the current active experiment, as well as its
        active recorder with the specified `status` type. Here is the example code of the method:

        .. code-block:: Python

            R.start_exp(experiment_name='test')
            ... # further operations
            R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)

        Parameters
        ----------
        status : str
            The status of a recorder, which can be SCHEDULED, RUNNING, FINISHED, FAILED.
        """
        self.exp_manager.end_exp(recorder_status)

    def search_records(self, experiment_ids, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria.

        The arguments of this function are not set to be rigid, and they will be different with different implementation of
        ``ExpManager`` in ``Qlib``. ``Qlib`` now provides an implementation of ``ExpManager`` with mlflow, and here is the
        example code of the method with the ``MLflowExpManager``:

        .. code-block:: Python

            R.log_metrics(m=2.50, step=0)
            records = R.search_records([experiment_id], order_by=["metrics.m DESC"])

        Parameters
        ----------
        experiment_ids : list
            list of experiment IDs.
        filter_string : str
            filter query string, defaults to searching all runs.
        run_view_type : int
            one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL (e.g. in mlflow.entities.ViewType).
        max_results  : int
            the maximum number of runs to put in the dataframe.
        order_by : list
            list of columns to order by (e.g., “metrics.rmse”).

        Returns
        -------
        A pandas.DataFrame of records, where each metric, parameter, and tag
        are expanded into their own columns named metrics.*, params.*, and tags.*
        respectively. For records that don't have a particular metric, parameter, or tag, their
        value will be (NumPy) Nan, None, or None respectively.
        """
        return self.exp_manager.search_records(experiment_ids, **kwargs)

    def list_experiments(self):
        """
        Method for listing all the existing experiments (except for those being deleted.)

        .. code-block:: Python

            exps = R.list_experiments()

        Returns
        -------
        A dictionary (name -> experiment) of experiments information that being stored.
        """
        return self.exp_manager.list_experiments()

    def list_recorders(self, experiment_id=None, experiment_name=None):
        """
        列出指定ID或名称实验的所有记录器的方法。

        如果用户未提供实验ID或名称，此方法将尝试获取默认实验并列出其所有记录器。
        如果默认实验不存在，方法将先创建默认实验，然后在其下创建新记录器。
        (关于默认实验的更多信息请参考`此处 <../component/recorder.html#qlib.workflow.exp.Experiment>`__)。

        示例代码：

        .. code-block:: Python

            recorders = R.list_recorders(experiment_name='test')

        参数
        ----------
        experiment_id : str
            实验ID
        experiment_name : str
            实验名称

        返回
        -------
        存储的记录器信息字典(id -> recorder)
        """
        return self.get_exp(experiment_id=experiment_id, experiment_name=experiment_name).list_recorders()

    def get_exp(
        self, *, experiment_id=None, experiment_name=None, create: bool = True, start: bool = False
    ) -> Experiment:
        """
        根据ID或名称获取实验的方法。当`create`参数设为True时，如果找不到有效实验，此方法将自动创建一个；否则仅获取特定实验或抛出错误。

        - 当'`create`'为True时：

            - 存在`active experiment`：
                - 未指定ID或名称，返回当前活动实验
                - 指定了ID或名称，返回指定实验。若无此实验，则创建新实验

            - 不存在`active experiment`：
                - 未指定ID或名称，创建默认实验并设为活动状态
                - 指定了ID或名称，返回指定实验。若无此实验，则创建新实验或默认实验

        - 当'`create`'为False时：

            - 存在`active experiment`：
                - 未指定ID或名称，返回当前活动实验
                - 指定了ID或名称，返回指定实验。若无此实验，抛出错误

            - 不存在`active experiment`：
                - 未指定ID或名称。若默认实验存在则返回，否则抛出错误
                - 指定了ID或名称，返回指定实验。若无此实验，抛出错误

        使用示例：

        .. code-block:: Python

            # 示例1
            with R.start('test'):
                exp = R.get_exp()
                recorders = exp.list_recorders()

            # 示例2
            with R.start('test'):
                exp = R.get_exp(experiment_name='test1')

            # 示例3
            exp = R.get_exp() -> 获取默认实验

            # 示例4
            exp = R.get_exp(experiment_name='test')

            # 示例5
            exp = R.get_exp(create=False) -> 若存在则返回默认实验

        参数
        ----------
        experiment_id : str
            实验ID
        experiment_name : str
            实验名称
        create : boolean
            决定当实验不存在时是否自动创建新实验
        start : bool
            当为True时，如果实验未启动(未激活)，将自动启动
            专为R.log_params自动启动实验设计

        返回
        -------
        具有给定ID或名称的实验实例
        """
        return self.exp_manager.get_exp(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            create=create,
            start=start,
        )

    def delete_exp(self, experiment_id=None, experiment_name=None):
        """
        删除指定ID或名称实验的方法。必须提供至少ID或名称中的一个，否则会出错。

        示例代码：

        .. code-block:: Python

            R.delete_exp(experiment_name='test')

        参数
        ----------
        experiment_id : str
            实验ID
        experiment_name : str
            实验名称
        """
        self.exp_manager.delete_exp(experiment_id, experiment_name)

    def get_uri(self):
        """
        获取当前实验管理器URI的方法。

        示例代码：

        .. code-block:: Python

            uri = R.get_uri()

        返回
        -------
        当前实验管理器的URI
        """
        return self.exp_manager.uri

    def set_uri(self, uri: Optional[Text]):
        """
        重置当前实验管理器**默认**URI的方法。

        注意：

        - 当URI引用文件路径时，请使用绝对路径而非类似"~/mlruns/"的字符串
          后端不支持此类字符串
        """
        self.exp_manager.default_uri = uri

    @contextmanager
    def uri_context(self, uri: Text):
        """
        临时设置exp_manager的**default_uri**为指定URI

        注意：
        - 请参考`set_uri`中的注意事项

        参数
        ----------
        uri : Text
            临时URI
        """
        prev_uri = self.exp_manager.default_uri
        self.exp_manager.default_uri = uri
        try:
            yield
        finally:
            self.exp_manager.default_uri = prev_uri

    def get_recorder(
        self,
        *,
        recorder_id=None,
        recorder_name=None,
        experiment_id=None,
        experiment_name=None,
    ) -> Recorder:
        """
        获取记录器的方法。

        - 存在`active recorder`时：

            - 未指定ID或名称，返回当前活动记录器

            - 指定了ID或名称，返回指定记录器

        - 不存在`active recorder`时：

            - 未指定ID或名称，抛出错误

            - 指定了ID或名称，必须同时提供对应的experiment_name才能返回指定记录器，否则抛出错误

        获取的记录器可用于后续操作如`save_object`、`load_object`、`log_params`、
        `log_metrics`等。

        使用示例：

        .. code-block:: Python

            # 示例1
            with R.start(experiment_name='test'):
                recorder = R.get_recorder()

            # 示例2
            with R.start(experiment_name='test'):
                recorder = R.get_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d')

            # 示例3
            recorder = R.get_recorder() -> 错误

            # 示例4
            recorder = R.get_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d') -> 错误

            # 示例5
            recorder = R.get_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d', experiment_name='test')


        用户可能关心的问题：
        - 问：如果多个记录器符合查询条件(如使用experiment_name查询)，会返回哪个记录器？
        - 答：如果使用mlflow后端，将返回具有最新`start_time`的记录器。因为MLflow的`search_runs`函数保证了这一点

        参数
        ----------
        recorder_id : str
            记录器ID
        recorder_name : str
            记录器名称
        experiment_name : str
            实验名称

        返回
        -------
        记录器实例
        """
        return self.get_exp(experiment_name=experiment_name, experiment_id=experiment_id, create=False).get_recorder(
            recorder_id, recorder_name, create=False, start=False
        )

    def delete_recorder(self, recorder_id=None, recorder_name=None):
        """
        删除指定ID或名称记录器的方法。必须提供至少ID或名称中的一个，否则会出错。

        示例代码：

        .. code-block:: Python

            R.delete_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d')

        参数
        ----------
        recorder_id : str
            记录器ID
        recorder_name : str
            记录器名称
        """
        self.get_exp().delete_recorder(recorder_id, recorder_name)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs: Dict[Text, Any]):
        """
        Method for saving objects as artifacts in the experiment to the uri. It supports either saving
        from a local file/directory, or directly saving objects. User can use valid python's keywords arguments
        to specify the object to be saved as well as its name (name: value).

        In summary, this API is designs for saving **objects** to **the experiments management backend path**,
        1. Qlib provide two methods to specify **objects**
        - Passing in the object directly by passing with `**kwargs` (e.g. R.save_objects(trained_model=model))
        - Passing in the local path to the object, i.e. `local_path` parameter.
        2. `artifact_path` represents the  **the experiments management backend path**

        - If `active recorder` exists: it will save the objects through the active recorder.
        - If `active recorder` not exists: the system will create a default experiment, and a new recorder and save objects under it.

        .. note::

            If one wants to save objects with a specific recorder. It is recommended to first get the specific recorder through `get_recorder` API and use the recorder the save objects. The supported arguments are the same as this method.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start(experiment_name='test'):
                pred = model.predict(dataset)
                R.save_objects(**{"pred.pkl": pred}, artifact_path='prediction')
                rid = R.get_recorder().id
            ...
            R.get_recorder(recorder_id=rid).load_object("prediction/pred.pkl")  #  after saving objects, you can load the previous object with this api

            # Case 2
            with R.start(experiment_name='test'):
                R.save_objects(local_path='results/pred.pkl', artifact_path="prediction")
                rid = R.get_recorder().id
            ...
            R.get_recorder(recorder_id=rid).load_object("prediction/pred.pkl")  #  after saving objects, you can load the previous object with this api


        Parameters
        ----------
        local_path : str
            if provided, them save the file or directory to the artifact URI.
        artifact_path : str
            the relative path for the artifact to be stored in the URI.
        **kwargs: Dict[Text, Any]
            the object to be saved.
            For example, `{"pred.pkl": pred}`
        """
        if local_path is not None and len(kwargs) > 0:
            raise ValueError(
                "You can choose only one of `local_path`(save the files in a path) or `kwargs`(pass in the objects directly)"
            )
        self.get_exp().get_recorder(start=True).save_objects(local_path, artifact_path, **kwargs)

    def load_object(self, name: Text):
        """
        从URI中实验的artifacts加载对象的方法。
        """
        return self.get_exp().get_recorder(start=True).load_object(name)

    def log_params(self, **kwargs):
        """
        在实验过程中记录参数的方法。除了使用``R``外，也可以通过`get_recorder`API获取特定记录器后进行记录。

        - 存在`active recorder`：通过活动记录器记录参数
        - 不存在`active recorder`：系统将创建默认实验和新记录器，并在其下记录参数

        使用示例：

        .. code-block:: Python

            # 示例1
            with R.start('test'):
                R.log_params(learning_rate=0.01)

            # 示例2
            R.log_params(learning_rate=0.01)

        参数
        ----------
        关键字参数：
            name1=value1, name2=value2, ...
        """
        self.get_exp(start=True).get_recorder(start=True).log_params(**kwargs)

    def log_metrics(self, step=None, **kwargs):
        """
        在实验过程中记录指标的方法。除了使用``R``外，也可以通过`get_recorder`API获取特定记录器后进行记录。

        - 存在`active recorder`：通过活动记录器记录指标
        - 不存在`active recorder`：系统将创建默认实验和新记录器，并在其下记录指标

        使用示例：

        .. code-block:: Python

            # 示例1
            with R.start('test'):
                R.log_metrics(train_loss=0.33, step=1)

            # 示例2
            R.log_metrics(train_loss=0.33, step=1)

        参数
        ----------
        关键字参数：
            name1=value1, name2=value2, ...
        """
        self.get_exp(start=True).get_recorder(start=True).log_metrics(step, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        将本地文件或目录记录为当前活动运行的artifact

        - 存在`active recorder`：通过活动记录器设置标签
        - 不存在`active recorder`：系统将创建默认实验和新记录器，并在其下设置标签

        参数
        ----------
        local_path : str
            要写入的文件路径
        artifact_path : Optional[str]
            如果提供，则为``artifact_uri``中要写入的目录
        """
        self.get_exp(start=True).get_recorder(start=True).log_artifact(local_path, artifact_path)

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        """
        Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.

        Parameters
        ----------
        path : str
            Relative source path to the desired artifact.
        dst_path : Optional[str]
            Absolute path of the local filesystem destination directory to which to
            download the specified artifacts. This directory must already exist.
            If unspecified, the artifacts will either be downloaded to a new
            uniquely-named directory on the local filesystem.

        Returns
        -------
        str
            Local path of desired artifact.
        """
        self.get_exp(start=True).get_recorder(start=True).download_artifact(path, dst_path)

    def set_tags(self, **kwargs):
        """
        为记录器设置标签的方法。除了使用``R``外，也可以通过`get_recorder`API获取特定记录器后进行设置。

        - 存在`active recorder`：通过活动记录器设置标签
        - 不存在`active recorder`：系统将创建默认实验和新记录器，并在其下设置标签

        使用示例：

        .. code-block:: Python

            # 示例1
            with R.start('test'):
                R.set_tags(release_version="2.2.0")

            # 示例2
            R.set_tags(release_version="2.2.0")

        参数
        ----------
        关键字参数：
            name1=value1, name2=value2, ...
        """
        self.get_exp(start=True).get_recorder(start=True).set_tags(**kwargs)


class RecorderWrapper(Wrapper):
    """
    QlibRecorder的包装类，用于检测用户在已经开始实验时是否重新初始化了qlib。
    """

    def register(self, provider):
        if self._provider is not None:
            expm = getattr(self._provider, "exp_manager")
            if expm.active_experiment is not None:
                raise RecorderInitializationError(
                    "Please don't reinitialize Qlib if QlibRecorder is already activated. Otherwise, the experiment stored location will be modified."
                )
        self._provider = provider


import sys

if sys.version_info >= (3, 9):
    from typing import Annotated

    QlibRecorderWrapper = Annotated[QlibRecorder, RecorderWrapper]
else:
    QlibRecorderWrapper = QlibRecorder

# global record
R: QlibRecorderWrapper = RecorderWrapper()
