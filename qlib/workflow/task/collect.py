# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
收集器模块可以从各处收集对象并进行处理，例如合并、分组、平均等操作。
"""

from collections import defaultdict
from qlib.log import TimeInspector
from typing import Callable, Dict, Iterable, List
from qlib.log import get_module_logger
from qlib.utils.serial import Serializable
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow import R
from qlib.workflow.exp import Experiment
from qlib.workflow.recorder import Recorder


class Collector(Serializable):
    """用于收集不同结果的收集器"""

    pickle_backend = "dill"  # use dill to dump user method

    def __init__(self, process_list=[]):
        """
        初始化收集器

        参数:
            process_list (list or Callable):  处理字典的处理器列表或单个处理器实例
        """
        if not isinstance(process_list, list):
            process_list = [process_list]
        self.process_list = process_list

    def collect(self) -> dict:
        """
        收集结果并返回一个类似{key: 值}的字典

        返回:
            dict: 收集后的字典

            例如:

            {"prediction": pd.Series}

            {"IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}

            ...
        """
        raise NotImplementedError(f"Please implement the `collect` method.")

    @staticmethod
    def process_collect(collected_dict, process_list=[], *args, **kwargs) -> dict:
        """
        对collect返回的字典进行一系列处理并返回一个类似{key: 值}的字典
        例如可以进行分组和集成

        参数:
            collected_dict (dict): collect方法返回的字典
            process_list (list or Callable): 处理字典的处理器列表或单个处理器实例
                处理器顺序与列表顺序相同
                例如: [Group1(..., Ensemble1()), Group2(..., Ensemble2())]

        返回:
            dict: 处理后的字典
        """
        if not isinstance(process_list, list):
            process_list = [process_list]
        result = {}
        for artifact in collected_dict:
            value = collected_dict[artifact]
            for process in process_list:
                if not callable(process):
                    raise NotImplementedError(f"{type(process)} is not supported in `process_collect`.")
                value = process(value, *args, **kwargs)
            result[artifact] = value
        return result

    def __call__(self, *args, **kwargs) -> dict:
        """
        执行包括``collect``和``process_collect``的工作流程

        返回:
            dict: 收集和处理后的字典
        """
        collected = self.collect()
        return self.process_collect(collected, self.process_list, *args, **kwargs)


class MergeCollector(Collector):
    """
    用于收集其他收集器结果的收集器

    例如:

        我们有两个收集器A和B
        A可以收集{"prediction": pd.Series}，B可以收集{"IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}
        经过本类收集后，我们可以收集{"A_prediction": pd.Series, "B_IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}

        ...

    """

    def __init__(self, collector_dict: Dict[str, Collector], process_list: List[Callable] = [], merge_func=None):
        """
        初始化MergeCollector

        参数:
            collector_dict (Dict[str,Collector]): 类似{collector_key, Collector}的字典
            process_list (List[Callable]): 处理字典的处理器列表或单个处理器实例
            merge_func (Callable): 生成最外层键的方法。参数是collector_dict中的``collector_key``和每个收集器收集后的``key``
                如果为None则使用元组连接它们，例如"ABC"+("a","b") -> ("ABC", ("a","b"))
        """
        super().__init__(process_list=process_list)
        self.collector_dict = collector_dict
        self.merge_func = merge_func

    def collect(self) -> dict:
        """
        收集collector_dict中的所有结果并将最外层键改为重组后的键

        返回:
            dict: 收集后的字典
        """
        collect_dict = {}
        for collector_key, collector in self.collector_dict.items():
            tmp_dict = collector()
            for key, value in tmp_dict.items():
                if self.merge_func is not None:
                    collect_dict[self.merge_func(collector_key, key)] = value
                else:
                    collect_dict[(collector_key, key)] = value
        return collect_dict


class RecorderCollector(Collector):
    ART_KEY_RAW = "__raw"

    def __init__(
        self,
        experiment,
        process_list=[],
        rec_key_func=None,
        rec_filter_func=None,
        artifacts_path={"pred": "pred.pkl"},
        artifacts_key=None,
        list_kwargs={},
        status: Iterable = {Recorder.STATUS_FI},
    ):
        """
        初始化RecorderCollector。

        参数：
            experiment:
                (Experiment或str): Experiment实例或Experiment名称
                (Callable): 可调用函数，返回实验列表
            process_list (list或Callable): 处理器列表或处理字典的处理器实例
            rec_key_func (Callable): 获取记录器键的函数。如果为None，则使用记录器ID
            rec_filter_func (Callable, 可选): 通过返回True或False过滤记录器。默认为None
            artifacts_path (dict, 可选): 记录器中工件名称及其路径。默认为{"pred": "pred.pkl"}
            artifacts_key (str或List, 可选): 要获取的工件键。如果为None，则获取所有工件
            list_kwargs (str): list_recorders函数的参数
            status (Iterable): 仅收集具有特定状态的记录器。None表示收集所有记录器
        """
        super().__init__(process_list=process_list)
        if isinstance(experiment, str):
            experiment = R.get_exp(experiment_name=experiment)
        assert isinstance(experiment, (Experiment, Callable))
        self.experiment = experiment
        self.artifacts_path = artifacts_path
        if rec_key_func is None:

            def rec_key_func(rec):
                return rec.info["id"]

        if artifacts_key is None:
            artifacts_key = list(self.artifacts_path.keys())
        self.rec_key_func = rec_key_func
        self.artifacts_key = artifacts_key
        self.rec_filter_func = rec_filter_func
        self.list_kwargs = list_kwargs
        self.status = status

    def collect(self, artifacts_key=None, rec_filter_func=None, only_exist=True) -> dict:
        """
        基于过滤后的记录器收集不同的工件。

        参数：
            artifacts_key (str或List, 可选): 要获取的工件键。如果为None，则使用默认值
            rec_filter_func (Callable, 可选): 通过返回True或False过滤记录器。如果为None，则使用默认值
            only_exist (bool, 可选): 是否仅当记录器确实拥有时才收集工件。
                如果为True，加载时出现异常的记录器将不会被收集。但如果为False，则会引发异常

        返回：
            dict: 收集后的字典，格式为{artifact: {rec_key: object}}
        """
        if artifacts_key is None:
            artifacts_key = self.artifacts_key
        if rec_filter_func is None:
            rec_filter_func = self.rec_filter_func

        if isinstance(artifacts_key, str):
            artifacts_key = [artifacts_key]

        collect_dict = {}
        # filter records

        if isinstance(self.experiment, Experiment):
            with TimeInspector.logt("Time to `list_recorders` in RecorderCollector"):
                recs = list(self.experiment.list_recorders(**self.list_kwargs).values())
        elif isinstance(self.experiment, Callable):
            recs = self.experiment()

        recs = [
            rec
            for rec in recs
            if (
                (self.status is None or rec.status in self.status) and (rec_filter_func is None or rec_filter_func(rec))
            )
        ]

        logger = get_module_logger("RecorderCollector")
        status_stat = defaultdict(int)
        for r in recs:
            status_stat[r.status] += 1
        logger.info(f"Nubmer of recorders after filter: {status_stat}")
        for rec in recs:
            rec_key = self.rec_key_func(rec)
            for key in artifacts_key:
                if self.ART_KEY_RAW == key:
                    artifact = rec
                else:
                    try:
                        artifact = rec.load_object(self.artifacts_path[key])
                    except LoadObjectError as e:
                        if only_exist:
                            # only collect existing artifact
                            logger.warning(f"Fail to load {self.artifacts_path[key]} and it is ignored.")
                            continue
                        raise e
                # give user some warning if the values are overridden
                cdd = collect_dict.setdefault(key, {})
                if rec_key in cdd:
                    logger.warning(
                        f"key '{rec_key}' is duplicated. Previous value will be overrides. Please check you `rec_key_func`"
                    )
                cdd[rec_key] = artifact

        return collect_dict

    def get_exp_name(self) -> str:
        """
        获取实验名称

        返回：
            str: 实验名称
        """
        return self.experiment.name
