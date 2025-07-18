# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TaskManager可以自动获取未使用的任务，并通过错误处理管理一组任务的生命周期。
这些功能可以并发运行任务，并确保每个任务只被使用一次。
Task Manager会将所有任务存储在`MongoDB <https://www.mongodb.com/>`_中。
使用此模块时，用户**必须**完成`MongoDB <https://www.mongodb.com/>`_的配置。

TaskManager中的任务由三部分组成
- 任务描述：定义任务内容
- 任务状态：任务的当前状态
- 任务结果：用户可以通过任务描述和任务结果获取任务
"""
import concurrent
import pickle
import time
from contextlib import contextmanager
from typing import Callable, List

import fire
import pymongo
from bson.binary import Binary
from bson.objectid import ObjectId
from pymongo.errors import InvalidDocument
from qlib import auto_init, get_module_logger
from tqdm.cli import tqdm

from .utils import get_mongodb
from ...config import C


class TaskManager:
    """
    TaskManager

    以下是TaskManager创建的任务示例：

    .. code-block:: python

        {
            'def': 'pickle序列化的任务定义，使用pickle更方便',
            'filter': '类JSON数据，用于过滤任务',
            'status': 'waiting' | 'running' | 'done',
            'res': 'pickle序列化的任务结果'
        }

    任务管理器假设您只会更新已获取的任务。
    MongoDB的获取和更新操作确保数据更新安全。

    此类可作为命令行工具使用。以下是几个示例：
    查看manage模块帮助的命令：
    python -m qlib.workflow.task.manage -h # 显示manage模块CLI手册
    python -m qlib.workflow.task.manage wait -h # 显示wait命令手册

    .. code-block:: shell

        python -m qlib.workflow.task.manage -t <pool_name> wait
        python -m qlib.workflow.task.manage -t <pool_name> task_stat


    .. note::

        假设：MongoDB中的数据会被编码，取出的数据会被解码

    四种状态说明：

        STATUS_WAITING: 等待训练

        STATUS_RUNNING: 训练中

        STATUS_PART_DONE: 已完成部分步骤，等待下一步

        STATUS_DONE: 全部工作完成
    """

    STATUS_WAITING = "waiting"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_PART_DONE = "part_done"

    ENCODE_FIELDS_PREFIX = ["def", "res"]

    def __init__(self, task_pool: str):
        """
        Init Task Manager, remember to make the statement of MongoDB url and database name firstly.
        A TaskManager instance serves a specific task pool.
        The static method of this module serves the whole MongoDB.

        Parameters
        ----------
        task_pool: str
            the name of Collection in MongoDB
        """
        self.task_pool: pymongo.collection.Collection = getattr(get_mongodb(), task_pool)
        self.logger = get_module_logger(self.__class__.__name__)
        self.logger.info(f"task_pool:{task_pool}")

    @staticmethod
    def list() -> list:
        """
        列出数据库中所有集合(任务池)。

        返回:
            list
        """
        return get_mongodb().list_collection_names()

    def _encode_task(self, task):
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = Binary(pickle.dumps(task[k], protocol=C.dump_protocol_version))
        return task

    def _decode_task(self, task):
        """
        _decode_task是序列化工具。
        Mongodb需要JSON格式，因此需要通过pickle将Python对象转换为JSON对象

        参数
        ----------
        task : dict
            任务信息

        返回
        -------
        dict
            mongodb所需的JSON格式
        """
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = pickle.loads(task[k])
        return task

    def _dict_to_str(self, flt):
        return {k: str(v) for k, v in flt.items()}

    def _decode_query(self, query):
        """
        如果查询包含`_id`，则需要使用`ObjectId`进行解码。
        例如，当使用TrainerRM时，需要查询`{"_id": {"$in": _id_list}}`，然后需要将`_id_list`中的每个`_id`转换为`ObjectId`。

        参数:
            query (dict): 查询字典，默认为{}

        返回:
            dict: 解码后的查询
        """
        if "_id" in query:
            if isinstance(query["_id"], dict):
                for key in query["_id"]:
                    query["_id"][key] = [ObjectId(i) for i in query["_id"][key]]
            else:
                query["_id"] = ObjectId(query["_id"])
        return query

    def replace_task(self, task, new_task):
        """
        Use a new task to replace a old one

        Args:
            task: old task
            new_task: new task
        """
        new_task = self._encode_task(new_task)
        query = {"_id": ObjectId(task["_id"])}
        try:
            self.task_pool.replace_one(query, new_task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            self.task_pool.replace_one(query, new_task)

    def insert_task(self, task):
        """
        Insert a task.

        Args:
            task: the task waiting for insert

        Returns:
            pymongo.results.InsertOneResult
        """
        try:
            insert_result = self.task_pool.insert_one(task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            insert_result = self.task_pool.insert_one(task)
        return insert_result

    def insert_task_def(self, task_def):
        """
        Insert a task to task_pool

        Parameters
        ----------
        task_def: dict
            the task definition

        Returns
        -------
        pymongo.results.InsertOneResult
        """
        task = self._encode_task(
            {
                "def": task_def,
                "filter": task_def,  # FIXME: catch the raised error
                "status": self.STATUS_WAITING,
            }
        )
        insert_result = self.insert_task(task)
        return insert_result

    def create_task(self, task_def_l, dry_run=False, print_nt=False) -> List[str]:
        """
        如果task_def_l中的任务是新的，则插入新任务到任务池并记录inserted_id。
        如果任务已存在，则只查询其_id。

        参数
        ----------
        task_def_l: list
            任务列表
        dry_run: bool
            是否实际插入新任务到任务池
        print_nt: bool
            是否打印新任务

        返回
        -------
        List[str]
            task_def_l中各任务的_id列表
        """
        new_tasks = []
        _id_list = []
        for t in task_def_l:
            try:
                r = self.task_pool.find_one({"filter": t})
            except InvalidDocument:
                r = self.task_pool.find_one({"filter": self._dict_to_str(t)})
            # When r is none, it indicates that r s a new task
            if r is None:
                new_tasks.append(t)
                if not dry_run:
                    insert_result = self.insert_task_def(t)
                    _id_list.append(insert_result.inserted_id)
                else:
                    _id_list.append(None)
            else:
                _id_list.append(self._decode_task(r)["_id"])

        self.logger.info(f"Total Tasks: {len(task_def_l)}, New Tasks: {len(new_tasks)}")

        if print_nt:  # print new task
            for t in new_tasks:
                print(t)

        if dry_run:
            return []

        return _id_list

    def fetch_task(self, query={}, status=STATUS_WAITING) -> dict:
        """
        使用查询获取任务。

        参数:
            query (dict, optional): 查询字典，默认为{}
            status (str, optional): 任务状态，默认为STATUS_WAITING

        返回:
            dict: 解码后的任务(集合中的文档)
        """
        query = query.copy()
        query = self._decode_query(query)
        query.update({"status": status})
        task = self.task_pool.find_one_and_update(
            query, {"$set": {"status": self.STATUS_RUNNING}}, sort=[("priority", pymongo.DESCENDING)]
        )
        # null will be at the top after sorting when using ASCENDING, so the larger the number higher, the higher the priority
        if task is None:
            return None
        task["status"] = self.STATUS_RUNNING
        return self._decode_task(task)

    @contextmanager
    def safe_fetch_task(self, query={}, status=STATUS_WAITING):
        """
        使用contextmanager从任务池中获取任务

        参数
        ----------
        query: dict
            查询字典

        返回
        -------
        dict: 解码后的任务(集合中的文档)
        """
        task = self.fetch_task(query=query, status=status)
        try:
            yield task
        except (Exception, KeyboardInterrupt):  # KeyboardInterrupt is not a subclass of Exception
            if task is not None:
                self.logger.info("Returning task before raising error")
                self.return_task(task, status=status)  # return task as the original status
                self.logger.info("Task returned")
            raise

    def task_fetcher_iter(self, query={}):
        while True:
            with self.safe_fetch_task(query=query) as task:
                if task is None:
                    break
                yield task

    def query(self, query={}, decode=True):
        """
        查询集合中的任务。
        如果迭代生成器耗时过长，此函数可能抛出异常`pymongo.errors.CursorNotFound: cursor id not found`

        示例:
             python -m qlib.workflow.task.manage -t <your task pool> query '{"_id": "615498be837d0053acbc5d58"}'

        参数
        ----------
        query: dict
            查询字典
        decode: bool
            是否解码结果

        返回
        -------
        dict: 解码后的任务(集合中的文档)
        """
        query = query.copy()
        query = self._decode_query(query)
        for t in self.task_pool.find(query):
            yield self._decode_task(t)

    def re_query(self, _id) -> dict:
        """
        使用_id查询任务。

        参数:
            _id (str): 文档的_id

        返回:
            dict: 解码后的任务(集合中的文档)
        """
        t = self.task_pool.find_one({"_id": ObjectId(_id)})
        return self._decode_task(t)

    def commit_task_res(self, task, res, status=STATUS_DONE):
        """
        提交结果到task['res']。

        参数:
            task ([type]): 任务
            res (object): 要保存的结果
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE。默认为STATUS_DONE。
        """
        # A workaround to use the class attribute.
        if status is None:
            status = TaskManager.STATUS_DONE
        self.task_pool.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": status, "res": Binary(pickle.dumps(res, protocol=C.dump_protocol_version))}},
        )

    def return_task(self, task, status=STATUS_WAITING):
        """
        Return a task to status. Always using in error handling.

        Args:
            task ([type]): [description]
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE. Defaults to STATUS_WAITING.
        """
        if status is None:
            status = TaskManager.STATUS_WAITING
        update_dict = {"$set": {"status": status}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def remove(self, query={}):
        """
        Remove the task using query

        Parameters
        ----------
        query: dict
            the dict of query

        """
        query = query.copy()
        query = self._decode_query(query)
        self.task_pool.delete_many(query)

    def task_stat(self, query={}) -> dict:
        """
        Count the tasks in every status.

        Args:
            query (dict, optional): the query dict. Defaults to {}.

        Returns:
            dict
        """
        query = query.copy()
        query = self._decode_query(query)
        tasks = self.query(query=query, decode=False)
        status_stat = {}
        for t in tasks:
            status_stat[t["status"]] = status_stat.get(t["status"], 0) + 1
        return status_stat

    def reset_waiting(self, query={}):
        """
        将所有运行中的任务重置为等待状态。可用于某些任务意外退出的情况。

        参数:
            query (dict, optional): 查询字典，默认为{}
        """
        query = query.copy()
        # default query
        if "status" not in query:
            query["status"] = self.STATUS_RUNNING
        return self.reset_status(query=query, status=self.STATUS_WAITING)

    def reset_status(self, query, status):
        query = query.copy()
        query = self._decode_query(query)
        print(self.task_pool.update_many(query, {"$set": {"status": status}}))

    def prioritize(self, task, priority: int):
        """
        Set priority for task

        Parameters
        ----------
        task : dict
            The task query from the database
        priority : int
            the target priority
        """
        update_dict = {"$set": {"priority": priority}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def _get_undone_n(self, task_stat):
        return (
            task_stat.get(self.STATUS_WAITING, 0)
            + task_stat.get(self.STATUS_RUNNING, 0)
            + task_stat.get(self.STATUS_PART_DONE, 0)
        )

    def _get_total(self, task_stat):
        return sum(task_stat.values())

    def wait(self, query={}):
        """
        在多进程环境下，主进程可能因为仍有任务在运行而无法从TaskManager获取任务。
        因此主进程应等待其他进程或机器完成所有任务。

        参数:
            query (dict, optional): 查询字典，默认为{}
        """
        task_stat = self.task_stat(query)
        total = self._get_total(task_stat)
        last_undone_n = self._get_undone_n(task_stat)
        if last_undone_n == 0:
            return
        self.logger.warning(f"Waiting for {last_undone_n} undone tasks. Please make sure they are running.")
        with tqdm(total=total, initial=total - last_undone_n) as pbar:
            while True:
                time.sleep(10)
                undone_n = self._get_undone_n(self.task_stat(query))
                pbar.update(last_undone_n - undone_n)
                last_undone_n = undone_n
                if undone_n == 0:
                    break

    def __str__(self):
        return f"TaskManager({self.task_pool})"


def run_task(
    task_func: Callable,
    task_pool: str,
    query: dict = {},
    force_release: bool = False,
    before_status: str = TaskManager.STATUS_WAITING,
    after_status: str = TaskManager.STATUS_DONE,
    **kwargs,
):
    r"""
    当任务池不为空(有WAITING状态任务)时，使用task_func获取并运行任务池中的任务

    运行此方法后，有以下4种情况(before_status -> after_status):

        STATUS_WAITING -> STATUS_DONE: 使用task["def"]作为`task_func`参数，表示任务尚未开始

        STATUS_WAITING -> STATUS_PART_DONE: 使用task["def"]作为`task_func`参数

        STATUS_PART_DONE -> STATUS_PART_DONE: use task["res"] as `task_func` param, it means that the task has been started but not completed

        STATUS_PART_DONE -> STATUS_DONE: use task["res"] as `task_func` param

    Parameters
    ----------
    task_func : Callable
        def (task_def, \**kwargs) -> <res which will be committed>

        the function to run the task
    task_pool : str
        the name of the task pool (Collection in MongoDB)
    query: dict
        will use this dict to query task_pool when fetching task
    force_release : bool
        will the program force to release the resource
    before_status : str:
        the tasks in before_status will be fetched and trained. Can be STATUS_WAITING, STATUS_PART_DONE.
    after_status : str:
        the tasks after trained will become after_status. Can be STATUS_WAITING, STATUS_PART_DONE.
    kwargs
        the params for `task_func`
    """
    tm = TaskManager(task_pool)

    ever_run = False

    while True:
        with tm.safe_fetch_task(status=before_status, query=query) as task:
            if task is None:
                break
            get_module_logger("run_task").info(task["def"])
            # when fetching `WAITING` task, use task["def"] to train
            if before_status == TaskManager.STATUS_WAITING:
                param = task["def"]
            # when fetching `PART_DONE` task, use task["res"] to train because the middle result has been saved to task["res"]
            elif before_status == TaskManager.STATUS_PART_DONE:
                param = task["res"]
            else:
                raise ValueError("The fetched task must be `STATUS_WAITING` or `STATUS_PART_DONE`!")
            if force_release:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    res = executor.submit(task_func, param, **kwargs).result()
            else:
                res = task_func(param, **kwargs)
            tm.commit_task_res(task, res, status=after_status)
            ever_run = True

    return ever_run


if __name__ == "__main__":
    # This is for using it in cmd
    # E.g. : `python -m qlib.workflow.task.manage list`
    auto_init()
    fire.Fire(TaskManager)
