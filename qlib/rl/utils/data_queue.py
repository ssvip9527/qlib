# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import multiprocessing
from multiprocessing.sharedctypes import Synchronized
import os
import threading
import time
import warnings
from queue import Empty
from typing import Any, Generator, Generic, Sequence, TypeVar, cast

from qlib.log import get_module_logger

_logger = get_module_logger(__name__)

T = TypeVar("T")

__all__ = ["DataQueue"]


class DataQueue(Generic[T]):
    """主进程(生产者)生成数据并存储在队列中。
    子进程(消费者)可以从队列中获取数据点。
    数据点通过从``dataset``读取项生成。

    :class:`DataQueue`是临时性的。当``repeat``耗尽时，
    必须创建一个新的DataQueue。

    更多背景请参阅:class:`qlib.rl.utils.FiniteVectorEnv`文档。

    参数
    ----------
    dataset
        从中读取数据的数据集。必须实现``__len__``和``__getitem__``。
    repeat
        数据点迭代次数。使用``-1``表示无限迭代。
    shuffle
        如果为True，项将以随机顺序读取。
    producer_num_workers
        数据加载的并发工作线程数。
    queue_maxsize
        队列阻塞前可放入的最大项数。

    示例
    --------
    >>> data_queue = DataQueue(my_dataset)
    >>> with data_queue:
    ...     ...

    在工作进程中:

    >>> for data in data_queue:
    ...     print(data)
    """

    def __init__(
        self,
        dataset: Sequence[T],
        repeat: int = 1,
        shuffle: bool = True,
        producer_num_workers: int = 0,
        queue_maxsize: int = 0,
    ) -> None:
        if queue_maxsize == 0:
            if os.cpu_count() is not None:
                queue_maxsize = cast(int, os.cpu_count())
                _logger.info(f"Automatically set data queue maxsize to {queue_maxsize} to avoid overwhelming.")
            else:
                queue_maxsize = 1
                _logger.warning(f"CPU count not available. Setting queue maxsize to 1.")

        self.dataset: Sequence[T] = dataset
        self.repeat: int = repeat
        self.shuffle: bool = shuffle
        self.producer_num_workers: int = producer_num_workers

        self._activated: bool = False
        self._queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=queue_maxsize)
        # Mypy 0.981 brought '"SynchronizedBase[Any]" has no attribute "value"  [attr-defined]' bug.
        # Therefore, add this type casting to pass Mypy checking.
        self._done = cast(Synchronized, multiprocessing.Value("i", 0))

    def __enter__(self) -> DataQueue:
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        with self._done.get_lock():
            self._done.value += 1
        for repeat in range(500):
            if repeat >= 1:
                warnings.warn(f"After {repeat} cleanup, the queue is still not empty.", category=RuntimeWarning)
            while not self._queue.empty():
                try:
                    self._queue.get(block=False)
                except Empty:
                    pass
            # Sometimes when the queue gets emptied, more data have already been sent,
            # and they are on the way into the queue.
            # If these data didn't get consumed, it will jam the queue and make the process hang.
            # We wait a second here for potential data arriving, and check again (for ``repeat`` times).
            time.sleep(1.0)
            if self._queue.empty():
                break
        _logger.debug(f"Remaining items in queue collection done. Empty: {self._queue.empty()}")

    def get(self, block: bool = True) -> Any:
        if not hasattr(self, "_first_get"):
            self._first_get = True
        if self._first_get:
            timeout = 5.0
            self._first_get = False
        else:
            timeout = 0.5
        while True:
            try:
                return self._queue.get(block=block, timeout=timeout)
            except Empty:
                if self._done.value:
                    raise StopIteration  # pylint: disable=raise-missing-from

    def put(self, obj: Any, block: bool = True, timeout: int | None = None) -> None:
        self._queue.put(obj, block=block, timeout=timeout)

    def mark_as_done(self) -> None:
        with self._done.get_lock():
            self._done.value = 1

    def done(self) -> int:
        return self._done.value

    def activate(self) -> DataQueue:
        if self._activated:
            raise ValueError("DataQueue can not activate twice.")
        thread = threading.Thread(target=self._producer, daemon=True)
        thread.start()
        self._activated = True
        return self

    def __del__(self) -> None:
        _logger.debug(f"__del__ of {__name__}.DataQueue")
        self.cleanup()

    def __iter__(self) -> Generator[Any, None, None]:
        if not self._activated:
            raise ValueError(
                "需要先调用activate()启动守护进程将数据放入队列才能使用。"
                "您可能忘记了在with块中使用DataQueue。"
            )
        return self._consumer()

    def _consumer(self) -> Generator[Any, None, None]:
        while True:
            try:
                yield self.get()
            except StopIteration:
                _logger.debug("Data consumer timed-out from get.")
                return

    def _producer(self) -> None:
        # pytorch dataloader is used here only because we need its sampler and multi-processing
        from torch.utils.data import DataLoader, Dataset  # pylint: disable=import-outside-toplevel

        try:
            dataloader = DataLoader(
                cast(Dataset[T], self.dataset),
                batch_size=None,
                num_workers=self.producer_num_workers,
                shuffle=self.shuffle,
                collate_fn=lambda t: t,  # identity collate fn
            )
            repeat = 10**18 if self.repeat == -1 else self.repeat
            for _rep in range(repeat):
                for data in dataloader:
                    if self._done.value:
                        # Already done.
                        return
                    self._queue.put(data)
                _logger.debug(f"Dataloader loop done. Repeat {_rep}.")
        finally:
            self.mark_as_done()
