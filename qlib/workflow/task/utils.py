# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Some tools for task management.
"""

import bisect
from copy import deepcopy
import pandas as pd
from qlib.data import D
from qlib.utils import hash_args
from qlib.utils.mod import init_instance_by_config
from qlib.workflow import R
from qlib.config import C
from qlib.log import get_module_logger
from pymongo import MongoClient
from pymongo.database import Database
from typing import Union
from pathlib import Path


def get_mongodb() -> Database:
    """
    获取MongoDB数据库实例，使用前需要先声明数据库地址和名称。

    示例:

        使用qlib.init():

            .. code-block:: python

                mongo_conf = {
                    "task_url": task_url,  # MongoDB地址
                    "task_db_name": task_db_name,  # 数据库名称
                }
                qlib.init(..., mongo=mongo_conf)

        在qlib.init()之后:

            .. code-block:: python

                C["mongo"] = {
                    "task_url" : "mongodb://localhost:27017/",
                    "task_db_name" : "rolling_db"
                }

    返回:
        Database: 数据库实例
    """
    try:
        cfg = C["mongo"]
    except KeyError:
        get_module_logger("task").error("Please configure `C['mongo']` before using TaskManager")
        raise
    get_module_logger("task").info(f"mongo config:{cfg}")
    client = MongoClient(cfg["task_url"])
    return client.get_database(name=cfg["task_db_name"])


def list_recorders(experiment, rec_filter_func=None):
    """
    列出实验中通过过滤器的所有记录器。

    参数:
        experiment (str or Experiment): 实验名称或实例
        rec_filter_func (Callable, optional): 返回True以保留给定记录器。默认为None。

    返回:
        dict: 过滤后的字典{rid: recorder}
    """
    if isinstance(experiment, str):
        experiment = R.get_exp(experiment_name=experiment)
    recs = experiment.list_recorders()
    recs_flt = {}
    for rid, rec in recs.items():
        if rec_filter_func is None or rec_filter_func(rec):
            recs_flt[rid] = rec

    return recs_flt


class TimeAdjuster:
    """
    Find appropriate date and adjust date.
    """

    def __init__(self, future=True, end_time=None):
        self._future = future
        self.cals = D.calendar(future=future, end_time=end_time)

    def set_end_time(self, end_time=None):
        """
        Set end time. None for use calendar's end time.

        Args:
            end_time
        """
        self.cals = D.calendar(future=self._future, end_time=end_time)

    def get(self, idx: int):
        """
        通过索引获取日期时间。

        参数
        ----------
        idx : int
            日历索引
        """
        if idx is None or idx >= len(self.cals):
            return None
        return self.cals[idx]

    def max(self) -> pd.Timestamp:
        """
        返回日历中的最大日期时间
        """
        return max(self.cals)

    def align_idx(self, time_point, tp_type="start") -> int:
        """
        对齐日历中时间点的索引。

        参数
        ----------
        time_point
        tp_type : str

        返回
        -------
        index : int
        """
        if time_point is None:
            # `None` indicates unbounded index/boarder
            return None
        time_point = pd.Timestamp(time_point)
        if tp_type == "start":
            idx = bisect.bisect_left(self.cals, time_point)
        elif tp_type == "end":
            idx = bisect.bisect_right(self.cals, time_point) - 1
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return idx

    def cal_interval(self, time_point_A, time_point_B) -> int:
        """
        Calculate the trading day interval (time_point_A - time_point_B)

        Args:
            time_point_A : time_point_A
            time_point_B : time_point_B (is the past of time_point_A)

        Returns:
            int: the interval between A and B
        """
        return self.align_idx(time_point_A) - self.align_idx(time_point_B)

    def align_time(self, time_point, tp_type="start") -> pd.Timestamp:
        """
        Align time_point to trade date of calendar

        Args:
            time_point
                Time point
            tp_type : str
                time point type (`"start"`, `"end"`)

        Returns:
            pd.Timestamp
        """
        if time_point is None:
            return None
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment: Union[dict, tuple]) -> Union[dict, tuple]:
        """
        Align the given date to the trade date

        for example:

            .. code-block:: python

                input: {'train': ('2008-01-01', '2014-12-31'), 'valid': ('2015-01-01', '2016-12-31'), 'test': ('2017-01-01', '2020-08-01')}

                output: {'train': (Timestamp('2008-01-02 00:00:00'), Timestamp('2014-12-31 00:00:00')),
                        'valid': (Timestamp('2015-01-05 00:00:00'), Timestamp('2016-12-30 00:00:00')),
                        'test': (Timestamp('2017-01-03 00:00:00'), Timestamp('2020-07-31 00:00:00'))}

        Parameters
        ----------
        segment

        Returns
        -------
        Union[dict, tuple]: the start and end trade date (pd.Timestamp) between the given start and end date.
        """
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, (tuple, list)):
            return self.align_time(segment[0], tp_type="start"), self.align_time(segment[1], tp_type="end")
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def truncate(self, segment: tuple, test_start, days: int) -> tuple:
        """
        Truncate the segment based on the test_start date

        Parameters
        ----------
        segment : tuple
            time segment
        test_start
        days : int
            The trading days to be truncated
            the data in this segment may need 'days' data
            `days` are based on the `test_start`.
            For example, if the label contains the information of 2 days in the near future, the prediction horizon 1 day.
            (e.g. the prediction target is `Ref($close, -2)/Ref($close, -1) - 1`)
            the days should be 2 + 1 == 3 days.

        Returns
        ---------
        tuple: new segment
        """
        test_idx = self.align_idx(test_start)
        if isinstance(segment, tuple):
            new_seg = []
            for time_point in segment:
                tp_idx = min(self.align_idx(time_point), test_idx - days)
                assert tp_idx > 0
                new_seg.append(self.get(tp_idx))
            return tuple(new_seg)
        else:
            raise NotImplementedError(f"This type of input is not supported")

    SHIFT_SD = "sliding"
    SHIFT_EX = "expanding"

    def _add_step(self, index, step):
        if index is None:
            return None
        return index + step

    def shift(self, seg: tuple, step: int, rtype=SHIFT_SD) -> tuple:
        """
        Shift the datetime of segment

        If there are None (which indicates unbounded index) in the segment, this method will return None.

        Parameters
        ----------
        seg :
            datetime segment
        step : int
            rolling step
        rtype : str
            rolling type ("sliding" or "expanding")

        Returns
        --------
        tuple: new segment

        Raises
        ------
        KeyError:
            shift will raise error if the index(both start and end) is out of self.cal
        """
        if isinstance(seg, tuple):
            start_idx, end_idx = self.align_idx(seg[0], tp_type="start"), self.align_idx(seg[1], tp_type="end")
            if rtype == self.SHIFT_SD:
                start_idx = self._add_step(start_idx, step)
                end_idx = self._add_step(end_idx, step)
            elif rtype == self.SHIFT_EX:
                end_idx = self._add_step(end_idx, step)
            else:
                raise NotImplementedError(f"This type of input is not supported")
            if start_idx is not None and start_idx > len(self.cals):
                raise KeyError("The segment is out of valid calendar")
            return self.get(start_idx), self.get(end_idx)
        else:
            raise NotImplementedError(f"This type of input is not supported")


def replace_task_handler_with_cache(task: dict, cache_dir: Union[str, Path] = ".") -> dict:
    """
    Replace the handler in task with a cache handler.
    It will automatically cache the file and save it in cache_dir.

    >>> import qlib
    >>> qlib.auto_init()
    >>> import datetime
    >>> # it is simplified task
    >>> task = {"dataset": {"kwargs":{'handler': {'class': 'Alpha158', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': datetime.date(2008, 1, 1), 'end_time': datetime.date(2020, 8, 1), 'fit_start_time': datetime.date(2008, 1, 1), 'fit_end_time': datetime.date(2014, 12, 31), 'instruments': 'CSI300'}}}}}
    >>> new_task = replace_task_handler_with_cache(task)
    >>> print(new_task)
    {'dataset': {'kwargs': {'handler': 'file...Alpha158.3584f5f8b4.pkl'}}}

    """
    cache_dir = Path(cache_dir)
    task = deepcopy(task)
    handler = task["dataset"]["kwargs"]["handler"]
    if isinstance(handler, dict):
        hash = hash_args(handler)
        h_path = cache_dir / f"{handler['class']}.{hash[:10]}.pkl"
        if not h_path.exists():
            h = init_instance_by_config(handler)
            h.to_pickle(h_path, dump_all=True)
        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    return task
