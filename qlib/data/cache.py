# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import sys
import stat
import time
import pickle
import traceback
import redis_lock
import contextlib
import abc
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, Iterable
from collections import OrderedDict

from ..config import C
from ..utils import (
    hash_args,
    get_redis_connection,
    read_bin,
    parse_field,
    remove_fields_space,
    normalize_cache_fields,
    normalize_cache_instruments,
)

from ..log import get_module_logger
from .base import Feature
from .ops import Operators  # pylint: disable=W0611  # noqa: F401


class QlibCacheException(RuntimeError):
    pass


class MemCacheUnit(abc.ABC):
    """内存缓存单元。"""

    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", 0)
        self._size = 0
        self.od = OrderedDict()

    def __setitem__(self, key, value):
        # TODO: 线程安全吗？__setitem__失败可能导致大小不一致？

        # 在od.__setitem__之后预先计算大小
        self._adjust_size(key, value)

        self.od.__setitem__(key, value)

        # 将键移至末尾，使其成为最新的
        self.od.move_to_end(key)

        if self.limited:
            # pop the oldest items beyond size limit
            while self._size > self.size_limit:
                self.popitem(last=False)

    def __getitem__(self, key):
        v = self.od.__getitem__(key)
        self.od.move_to_end(key)
        return v

    def __contains__(self, key):
        return key in self.od

    def __len__(self):
        return self.od.__len__()

    def __repr__(self):
        return f"{self.__class__.__name__}<size_limit:{self.size_limit if self.limited else 'no limit'} total_size:{self._size}>\n{self.od.__repr__()}"

    def set_limit_size(self, limit):
        self.size_limit = limit

    @property
    def limited(self):
        """内存缓存是否有限制"""
        return self.size_limit > 0

    @property
    def total_size(self):
        return self._size

    def clear(self):
        self._size = 0
        self.od.clear()

    def popitem(self, last=True):
        k, v = self.od.popitem(last=last)
        self._size -= self._get_value_size(v)

        return k, v

    def pop(self, key):
        v = self.od.pop(key)
        self._size -= self._get_value_size(v)

        return v

    def _adjust_size(self, key, value):
        if key in self.od:
            self._size -= self._get_value_size(self.od[key])

        self._size += self._get_value_size(value)

    @abc.abstractmethod
    def _get_value_size(self, value):
        raise NotImplementedError


class MemCacheLengthUnit(MemCacheUnit):
    def __init__(self, size_limit=0):
        super().__init__(size_limit=size_limit)

    def _get_value_size(self, value):
        return 1


class MemCacheSizeofUnit(MemCacheUnit):
    def __init__(self, size_limit=0):
        super().__init__(size_limit=size_limit)

    def _get_value_size(self, value):
        return sys.getsizeof(value)


class MemCache:
    """内存缓存。"""

    def __init__(self, mem_cache_size_limit=None, limit_type="length"):
        """

        参数
        ----------
        mem_cache_size_limit:
            缓存最大大小。
        limit_type:
            length或sizeof；length（调用函数：len），size（调用函数：sys.getsizeof）。
        """

        size_limit = C.mem_cache_size_limit if mem_cache_size_limit is None else mem_cache_size_limit
        limit_type = C.mem_cache_limit_type if limit_type is None else limit_type

        if limit_type == "length":
            klass = MemCacheLengthUnit
        elif limit_type == "sizeof":
            klass = MemCacheSizeofUnit
        else:
            raise ValueError(f"limit_type must be length or sizeof, your limit_type is {limit_type}")

        self.__calendar_mem_cache = klass(size_limit)
        self.__instrument_mem_cache = klass(size_limit)
        self.__feature_mem_cache = klass(size_limit)

    def __getitem__(self, key):
        if key == "c":
            return self.__calendar_mem_cache
        elif key == "i":
            return self.__instrument_mem_cache
        elif key == "f":
            return self.__feature_mem_cache
        else:
            raise KeyError("未知的内存缓存单元")

    def clear(self):
        self.__calendar_mem_cache.clear()
        self.__instrument_mem_cache.clear()
        self.__feature_mem_cache.clear()


class MemCacheExpire:
    CACHE_EXPIRE = C.mem_cache_expire

    @staticmethod
    def set_cache(mem_cache, key, value):
        """设置缓存

        :param mem_cache: MemCache属性('c'/'i'/'f')。
        :param key: 缓存键。
        :param value: 缓存值。
        """
        mem_cache[key] = value, time.time()

    @staticmethod
    def get_cache(mem_cache, key):
        """获取内存缓存

        :param mem_cache: MemCache属性('c'/'i'/'f')。
        :param key: 缓存键。
        :return: 缓存值；如果缓存不存在，返回None。
        """
        value = None
        expire = False
        if key in mem_cache:
            value, latest_time = mem_cache[key]
            expire = (time.time() - latest_time) > MemCacheExpire.CACHE_EXPIRE
        return value, expire


class CacheUtils:
    LOCK_ID = "QLIB"

    @staticmethod
    def organize_meta_file():
        pass

    @staticmethod
    def reset_lock():
        r = get_redis_connection()
        redis_lock.reset_all(r)

    @staticmethod
    def visit(cache_path: Union[str, Path]):
        # FIXME: Because read_lock was canceled when reading the cache, multiple processes may have read and write exceptions here
        try:
            cache_path = Path(cache_path)
            meta_path = cache_path.with_suffix(".meta")
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            with meta_path.open("wb") as f:
                try:
                    d["meta"]["last_visit"] = str(time.time())
                    d["meta"]["visits"] = d["meta"]["visits"] + 1
                except KeyError as key_e:
                    raise KeyError("Unknown meta keyword") from key_e
                pickle.dump(d, f, protocol=C.dump_protocol_version)
        except Exception as e:
            get_module_logger("CacheUtils").warning(f"visit {cache_path} cache error: {e}")

    @staticmethod
    def acquire(lock, lock_name):
        try:
            lock.acquire()
        except redis_lock.AlreadyAcquired as lock_acquired:
            raise QlibCacheException(
                f"""It sees the key(lock:{repr(lock_name)[1:-1]}-wlock) of the redis lock has existed in your redis db now.
                    You can use the following command to clear your redis keys and rerun your commands:
                    $ redis-cli
                    > select {C.redis_task_db}
                    > del "lock:{repr(lock_name)[1:-1]}-wlock"
                    > quit
                    If the issue is not resolved, use "keys *" to find if multiple keys exist. If so, try using "flushall" to clear all the keys.
                """
            ) from lock_acquired

    @staticmethod
    @contextlib.contextmanager
    def reader_lock(redis_t, lock_name: str):
        current_cache_rlock = redis_lock.Lock(redis_t, f"{lock_name}-rlock")
        current_cache_wlock = redis_lock.Lock(redis_t, f"{lock_name}-wlock")
        lock_reader = f"{lock_name}-reader"
        # make sure only one reader is entering
        current_cache_rlock.acquire(timeout=60)
        try:
            current_cache_readers = redis_t.get(lock_reader)
            if current_cache_readers is None or int(current_cache_readers) == 0:
                CacheUtils.acquire(current_cache_wlock, lock_name)
            redis_t.incr(lock_reader)
        finally:
            current_cache_rlock.release()
        try:
            yield
        finally:
            # make sure only one reader is leaving
            current_cache_rlock.acquire(timeout=60)
            try:
                redis_t.decr(lock_reader)
                if int(redis_t.get(lock_reader)) == 0:
                    redis_t.delete(lock_reader)
                    current_cache_wlock.reset()
            finally:
                current_cache_rlock.release()

    @staticmethod
    @contextlib.contextmanager
    def writer_lock(redis_t, lock_name):
        current_cache_wlock = redis_lock.Lock(redis_t, f"{lock_name}-wlock", id=CacheUtils.LOCK_ID)
        CacheUtils.acquire(current_cache_wlock, lock_name)
        try:
            yield
        finally:
            current_cache_wlock.release()


class BaseProviderCache:
    """Provider cache base class"""

    def __init__(self, provider):
        self.provider = provider
        self.logger = get_module_logger(self.__class__.__name__)

    def __getattr__(self, attr):
        return getattr(self.provider, attr)

    @staticmethod
    def check_cache_exists(cache_path: Union[str, Path], suffix_list: Iterable = (".index", ".meta")) -> bool:
        cache_path = Path(cache_path)
        for p in [cache_path] + [cache_path.with_suffix(_s) for _s in suffix_list]:
            if not p.exists():
                return False
        return True

    @staticmethod
    def clear_cache(cache_path: Union[str, Path]):
        for p in [
            cache_path,
            cache_path.with_suffix(".meta"),
            cache_path.with_suffix(".index"),
        ]:
            if p.exists():
                p.unlink()

    @staticmethod
    def get_cache_dir(dir_name: str, freq: str = None) -> Path:
        cache_dir = Path(C.dpm.get_data_uri(freq)).joinpath(dir_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


class ExpressionCache(BaseProviderCache):
    """Expression cache mechanism base class.

    This class is used to wrap expression provider with self-defined expression cache mechanism.

    .. note:: Override the `_uri` and `_expression` method to create your own expression cache mechanism.
    """

    def expression(self, instrument, field, start_time, end_time, freq):
        """Get expression data.

        .. note:: Same interface as `expression` method in expression provider
        """
        try:
            return self._expression(instrument, field, start_time, end_time, freq)
        except NotImplementedError:
            return self.provider.expression(instrument, field, start_time, end_time, freq)

    def _uri(self, instrument, field, start_time, end_time, freq):
        """Get expression cache file uri.

        Override this method to define how to get expression cache file uri corresponding to users' own cache mechanism.
        """
        raise NotImplementedError("Implement this function to match your own cache mechanism")

    def _expression(self, instrument, field, start_time, end_time, freq):
        """Get expression data using cache.

        Override this method to define how to get expression data corresponding to users' own cache mechanism.
        """
        raise NotImplementedError("Implement this method if you want to use expression cache")

    def update(self, cache_uri: Union[str, Path], freq: str = "day"):
        """Update expression cache to latest calendar.

        Override this method to define how to update expression cache corresponding to users' own cache mechanism.

        Parameters
        ----------
        cache_uri : str or Path
            the complete uri of expression cache file (include dir path).
        freq : str

        Returns
        -------
        int
            0(successful update)/ 1(no need to update)/ 2(update failure).
        """
        raise NotImplementedError("Implement this method if you want to make expression cache up to date")


class DatasetCache(BaseProviderCache):
    """Dataset cache mechanism base class.

    This class is used to wrap dataset provider with self-defined dataset cache mechanism.

    .. note:: Override the `_uri` and `_dataset` method to create your own dataset cache mechanism.
    """

    HDF_KEY = "df"

    def dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        """Get feature dataset.

        .. note:: Same interface as `dataset` method in dataset provider

        .. note:: The server use redis_lock to make sure
            read-write conflicts will not be triggered
            but client readers are not considered.
        """
        if disk_cache == 0:
            # skip cache
            return self.provider.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
            )
        else:
            # use and replace cache
            try:
                return self._dataset(
                    instruments, fields, start_time, end_time, freq, disk_cache, inst_processors=inst_processors
                )
            except NotImplementedError:
                return self.provider.dataset(
                    instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
                )

    def _uri(self, instruments, fields, start_time, end_time, freq, **kwargs):
        """Get dataset cache file uri.

        Override this method to define how to get dataset cache file uri corresponding to users' own cache mechanism.
        """
        raise NotImplementedError("Implement this function to match your own cache mechanism")

    def _dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        """Get feature dataset using cache.

        Override this method to define how to get feature dataset corresponding to users' own cache mechanism.
        """
        raise NotImplementedError("Implement this method if you want to use dataset feature cache")

    def _dataset_uri(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        """Get a uri of feature dataset using cache.
        specially:
            disk_cache=1 means using data set cache and return the uri of cache file.
            disk_cache=0 means client knows the path of expression cache,
                         server checks if the cache exists(if not, generate it), and client loads data by itself.
        Override this method to define how to get feature dataset uri corresponding to users' own cache mechanism.
        """
        raise NotImplementedError(
            "Implement this method if you want to use dataset feature cache as a cache file for client"
        )

    def update(self, cache_uri: Union[str, Path], freq: str = "day"):
        """Update dataset cache to latest calendar.

        Override this method to define how to update dataset cache corresponding to users' own cache mechanism.

        Parameters
        ----------
        cache_uri : str or Path
            the complete uri of dataset cache file (include dir path).
        freq : str

        Returns
        -------
        int
            0(successful update)/ 1(no need to update)/ 2(update failure)
        """
        raise NotImplementedError("Implement this method if you want to make expression cache up to date")

    @staticmethod
    def cache_to_origin_data(data, fields):
        """cache data to origin data

        :param data: pd.DataFrame, cache data.
        :param fields: feature fields.
        :return: pd.DataFrame.
        """
        not_space_fields = remove_fields_space(fields)
        data = data.loc[:, not_space_fields]
        # set features fields
        data.columns = [str(i) for i in fields]
        return data

    @staticmethod
    def normalize_uri_args(instruments, fields, freq):
        """normalize uri args"""
        instruments = normalize_cache_instruments(instruments)
        fields = normalize_cache_fields(fields)
        freq = freq.lower()

        return instruments, fields, freq


class DiskExpressionCache(ExpressionCache):
    """Prepared cache mechanism for server."""

    def __init__(self, provider, **kwargs):
        super(DiskExpressionCache, self).__init__(provider)
        self.r = get_redis_connection()
        # remote==True means client is using this module, writing behaviour will not be allowed.
        self.remote = kwargs.get("remote", False)

    def get_cache_dir(self, freq: str = None) -> Path:
        return super(DiskExpressionCache, self).get_cache_dir(C.features_cache_dir_name, freq)

    def _uri(self, instrument, field, start_time, end_time, freq):
        field = remove_fields_space(field)
        instrument = str(instrument).lower()
        return hash_args(instrument, field, freq)

    def _expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
        _cache_uri = self._uri(instrument=instrument, field=field, start_time=None, end_time=None, freq=freq)
        _instrument_dir = self.get_cache_dir(freq).joinpath(instrument.lower())
        cache_path = _instrument_dir.joinpath(_cache_uri)
        # get calendar
        from .data import Cal  # pylint: disable=C0415

        _calendar = Cal.calendar(freq=freq)

        _, _, start_index, end_index = Cal.locate_index(start_time, end_time, freq, future=False)

        if self.check_cache_exists(cache_path, suffix_list=[".meta"]):
            """
            In most cases, we do not need reader_lock.
            Because updating data is a small probability event compare to reading data.

            """
            # FIXME: Removing the reader lock may result in conflicts.
            # with CacheUtils.reader_lock(self.r, 'expression-%s' % _cache_uri):

            # modify expression cache meta file
            try:
                # FIXME: Multiple readers may result in error visit number
                if not self.remote:
                    CacheUtils.visit(cache_path)
                series = read_bin(cache_path, start_index, end_index)
                return series
            except Exception:
                series = None
                self.logger.error("reading %s file error : %s" % (cache_path, traceback.format_exc()))
            return series
        else:
            # normalize field
            field = remove_fields_space(field)
            # cache unavailable, generate the cache
            _instrument_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(eval(parse_field(field)), Feature):
                # When the expression is not a raw feature
                # generate expression cache if the feature is not a Feature
                # instance
                series = self.provider.expression(instrument, field, _calendar[0], _calendar[-1], freq)
                if not series.empty:
                    # This expression is empty, we don't generate any cache for it.
                    with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:expression-{_cache_uri}"):
                        self.gen_expression_cache(
                            expression_data=series,
                            cache_path=cache_path,
                            instrument=instrument,
                            field=field,
                            freq=freq,
                            last_update=str(_calendar[-1]),
                        )
                    return series.loc[start_index:end_index]
                else:
                    return series
            else:
                # If the expression is a raw feature(such as $close, $open)
                return self.provider.expression(instrument, field, start_time, end_time, freq)

    def gen_expression_cache(self, expression_data, cache_path, instrument, field, freq, last_update):
        """use bin file to save like feature-data."""
        # Make sure the cache runs right when the directory is deleted
        # while running
        meta = {
            "info": {"instrument": instrument, "field": field, "freq": freq, "last_update": last_update},
            "meta": {"last_visit": time.time(), "visits": 1},
        }
        self.logger.debug(f"generating expression cache: {meta}")
        self.clear_cache(cache_path)
        meta_path = cache_path.with_suffix(".meta")

        with meta_path.open("wb") as f:
            pickle.dump(meta, f, protocol=C.dump_protocol_version)
        meta_path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        df = expression_data.to_frame()

        r = np.hstack([df.index[0], expression_data]).astype("<f")
        r.tofile(str(cache_path))

    def update(self, sid, cache_uri, freq: str = "day"):
        cp_cache_uri = self.get_cache_dir(freq).joinpath(sid).joinpath(cache_uri)
        meta_path = cp_cache_uri.with_suffix(".meta")
        if not self.check_cache_exists(cp_cache_uri, suffix_list=[".meta"]):
            self.logger.info(f"The cache {cp_cache_uri} has corrupted. It will be removed")
            self.clear_cache(cp_cache_uri)
            return 2

        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri())}:expression-{cache_uri}"):
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            instrument = d["info"]["instrument"]
            field = d["info"]["field"]
            freq = d["info"]["freq"]
            last_update_time = d["info"]["last_update"]

            # get newest calendar
            from .data import Cal, ExpressionD  # pylint: disable=C0415

            whole_calendar = Cal.calendar(start_time=None, end_time=None, freq=freq)
            # calendar since last updated.
            new_calendar = Cal.calendar(start_time=last_update_time, end_time=None, freq=freq)

            # get append data
            if len(new_calendar) <= 1:
                # Including last updated calendar, we only get 1 item.
                # No future updating is needed.
                return 1
            else:
                # get the data needed after the historical data are removed.
                # The start index of new data
                current_index = len(whole_calendar) - len(new_calendar) + 1

                # The existing data length
                size_bytes = os.path.getsize(cp_cache_uri)
                ele_size = np.dtype("<f").itemsize
                assert size_bytes % ele_size == 0
                ele_n = size_bytes // ele_size - 1

                expr = ExpressionD.get_expression_instance(field)
                lft_etd, rght_etd = expr.get_extended_window_size()
                # The expression used the future data after rght_etd days.
                # So the last rght_etd data should be removed.
                # There are most `ele_n` period of data can be remove
                remove_n = min(rght_etd, ele_n)
                assert new_calendar[1] == whole_calendar[current_index]
                data = self.provider.expression(
                    instrument, field, whole_calendar[current_index - remove_n], new_calendar[-1], freq
                )
                with open(cp_cache_uri, "ab") as f:
                    data = np.array(data).astype("<f")
                    # Remove the last bits
                    f.truncate(size_bytes - ele_size * remove_n)
                    f.write(data)
                # update meta file
                d["info"]["last_update"] = str(new_calendar[-1])
                with meta_path.open("wb") as f:
                    pickle.dump(d, f, protocol=C.dump_protocol_version)
        return 0


class DiskDatasetCache(DatasetCache):
    """Prepared cache mechanism for server."""

    def __init__(self, provider, **kwargs):
        super(DiskDatasetCache, self).__init__(provider)
        self.r = get_redis_connection()
        self.remote = kwargs.get("remote", False)

    @staticmethod
    def _uri(instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        return hash_args(*DatasetCache.normalize_uri_args(instruments, fields, freq), disk_cache, inst_processors)

    def get_cache_dir(self, freq: str = None) -> Path:
        return super(DiskDatasetCache, self).get_cache_dir(C.dataset_cache_dir_name, freq)

    @classmethod
    def read_data_from_cache(cls, cache_path: Union[str, Path], start_time, end_time, fields):
        """read_cache_from

        This function can read data from the disk cache dataset

        :param cache_path:
        :param start_time:
        :param end_time:
        :param fields: The fields order of the dataset cache is sorted. So rearrange the columns to make it consistent.
        :return:
        """

        im = DiskDatasetCache.IndexManager(cache_path)
        index_data = im.get_index(start_time, end_time)
        if index_data.shape[0] > 0:
            start, stop = (
                index_data["start"].iloc[0].item(),
                index_data["end"].iloc[-1].item(),
            )
        else:
            start = stop = 0

        with pd.HDFStore(cache_path, mode="r") as store:
            if "/{}".format(im.KEY) in store.keys():
                df = store.select(key=im.KEY, start=start, stop=stop)
                df = df.swaplevel("datetime", "instrument").sort_index()
                # read cache and need to replace not-space fields to field
                df = cls.cache_to_origin_data(df, fields)

            else:
                df = pd.DataFrame(columns=fields)
        return df

    def _dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]
    ):
        if disk_cache == 0:
            # In this case, data_set cache is configured but will not be used.
            return self.provider.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
            )
        # FIXME: The cache after resample, when read again and intercepted with end_time, results in incomplete data date
        if inst_processors:
            raise ValueError(
                f"{self.__class__.__name__} does not support inst_processor. "
                f"Please use `D.features(disk_cache=0)` or `qlib.init(dataset_cache=None)`"
            )
        _cache_uri = self._uri(
            instruments=instruments,
            fields=fields,
            start_time=None,
            end_time=None,
            freq=freq,
            disk_cache=disk_cache,
            inst_processors=inst_processors,
        )

        cache_path = self.get_cache_dir(freq).joinpath(_cache_uri)

        features = pd.DataFrame()
        gen_flag = False

        if self.check_cache_exists(cache_path):
            if disk_cache == 1:
                # use cache
                with CacheUtils.reader_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                    CacheUtils.visit(cache_path)
                    features = self.read_data_from_cache(cache_path, start_time, end_time, fields)
            elif disk_cache == 2:
                gen_flag = True
        else:
            gen_flag = True

        if gen_flag:
            # cache unavailable, generate the cache
            with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                features = self.gen_dataset_cache(
                    cache_path=cache_path,
                    instruments=instruments,
                    fields=fields,
                    freq=freq,
                    inst_processors=inst_processors,
                )
            if not features.empty:
                features = features.sort_index().loc(axis=0)[:, start_time:end_time]
        return features

    def _dataset_uri(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]
    ):
        if disk_cache == 0:
            # In this case, server only checks the expression cache.
            # The client will load the cache data by itself.
            from .data import LocalDatasetProvider  # pylint: disable=C0415

            LocalDatasetProvider.multi_cache_walker(instruments, fields, start_time, end_time, freq)
            return ""
        # FIXME: The cache after resample, when read again and intercepted with end_time, results in incomplete data date
        if inst_processors:
            raise ValueError(
                f"{self.__class__.__name__} does not support inst_processor. "
                f"Please use `D.features(disk_cache=0)` or `qlib.init(dataset_cache=None)`"
            )
        _cache_uri = self._uri(
            instruments=instruments,
            fields=fields,
            start_time=None,
            end_time=None,
            freq=freq,
            disk_cache=disk_cache,
            inst_processors=inst_processors,
        )
        cache_path = self.get_cache_dir(freq).joinpath(_cache_uri)

        if self.check_cache_exists(cache_path):
            self.logger.debug(f"The cache dataset has already existed {cache_path}. Return the uri directly")
            with CacheUtils.reader_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                CacheUtils.visit(cache_path)
            return _cache_uri
        else:
            # cache unavailable, generate the cache
            with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                self.gen_dataset_cache(
                    cache_path=cache_path,
                    instruments=instruments,
                    fields=fields,
                    freq=freq,
                    inst_processors=inst_processors,
                )
            return _cache_uri

    class IndexManager:
        """
        The lock is not considered in the class. Please consider the lock outside the code.
        This class is the proxy of the disk data.
        """

        KEY = "df"

        def __init__(self, cache_path: Union[str, Path]):
            self.index_path = cache_path.with_suffix(".index")
            self._data = None
            self.logger = get_module_logger(self.__class__.__name__)

        def get_index(self, start_time=None, end_time=None):
            # TODO: fast read index from the disk.
            if self._data is None:
                self.sync_from_disk()
            return self._data.loc[start_time:end_time].copy()

        def sync_to_disk(self):
            if self._data is None:
                raise ValueError("No data to sync to disk.")
            self._data.sort_index(inplace=True)
            self._data.to_hdf(self.index_path, key=self.KEY, mode="w", format="table")
            # The index should be readable for all users
            self.index_path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        def sync_from_disk(self):
            # The file will not be closed directly if we read_hdf from the disk directly
            with pd.HDFStore(self.index_path, mode="r") as store:
                if "/{}".format(self.KEY) in store.keys():
                    self._data = pd.read_hdf(store, key=self.KEY)
                else:
                    self._data = pd.DataFrame()

        def update(self, data, sync=True):
            self._data = data.astype(np.int32).copy()
            if sync:
                self.sync_to_disk()

        def append_index(self, data, to_disk=True):
            data = data.astype(np.int32).copy()
            data.sort_index(inplace=True)
            self._data = pd.concat([self._data, data])
            if to_disk:
                with pd.HDFStore(self.index_path) as store:
                    store.append(self.KEY, data, append=True)

        @staticmethod
        def build_index_from_data(data, start_index=0):
            if data.empty:
                return pd.DataFrame()
            line_data = data.groupby("datetime", group_keys=False).size()
            line_data.sort_index(inplace=True)
            index_end = line_data.cumsum()
            index_start = index_end.shift(1, fill_value=0)

            index_data = pd.DataFrame()
            index_data["start"] = index_start
            index_data["end"] = index_end
            index_data += start_index
            return index_data

    def gen_dataset_cache(self, cache_path: Union[str, Path], instruments, fields, freq, inst_processors=[]):
        """gen_dataset_cache

        .. note:: This function does not consider the cache read write lock. Please
            acquire the lock outside this function

        The format the cache contains 3 parts(followed by typical filename).

        - index : cache/d41366901e25de3ec47297f12e2ba11d.index

            - The content of the file may be in following format(pandas.Series)

                .. code-block:: python

                                        start end
                    1999-11-10 00:00:00     0   1
                    1999-11-11 00:00:00     1   2
                    1999-11-12 00:00:00     2   3
                    ...

                .. note:: The start is closed. The end is open!!!!!

            - Each line contains two element <start_index, end_index> with a timestamp as its index.
            - It indicates the `start_index` (included) and `end_index` (excluded) of the data for `timestamp`

        - meta data: cache/d41366901e25de3ec47297f12e2ba11d.meta

        - data     : cache/d41366901e25de3ec47297f12e2ba11d

            - This is a hdf file sorted by datetime

        :param cache_path:  The path to store the cache.
        :param instruments:  The instruments to store the cache.
        :param fields:  The fields to store the cache.
        :param freq:  The freq to store the cache.
        :param inst_processors:  Instrument processors.

        :return type pd.DataFrame; The fields of the returned DataFrame are consistent with the parameters of the function.
        """
        # get calendar
        from .data import Cal  # pylint: disable=C0415

        cache_path = Path(cache_path)
        _calendar = Cal.calendar(freq=freq)
        self.logger.debug(f"Generating dataset cache {cache_path}")
        # Make sure the cache runs right when the directory is deleted
        # while running
        self.clear_cache(cache_path)

        features = self.provider.dataset(
            instruments, fields, _calendar[0], _calendar[-1], freq, inst_processors=inst_processors
        )

        if features.empty:
            return features

        # swap index and sorted
        features = features.swaplevel("instrument", "datetime").sort_index()

        # write cache data
        with pd.HDFStore(str(cache_path.with_suffix(".data"))) as store:
            cache_to_orig_map = dict(zip(remove_fields_space(features.columns), features.columns))
            orig_to_cache_map = dict(zip(features.columns, remove_fields_space(features.columns)))
            cache_features = features[list(cache_to_orig_map.values())].rename(columns=orig_to_cache_map)
            # cache columns
            cache_columns = sorted(cache_features.columns)
            cache_features = cache_features.loc[:, cache_columns]
            cache_features = cache_features.loc[:, ~cache_features.columns.duplicated()]
            store.append(DatasetCache.HDF_KEY, cache_features, append=False)
        # write meta file
        meta = {
            "info": {
                "instruments": instruments,
                "fields": list(cache_features.columns),
                "freq": freq,
                "last_update": str(_calendar[-1]),  # The last_update to store the cache
                "inst_processors": inst_processors,  # The last_update to store the cache
            },
            "meta": {"last_visit": time.time(), "visits": 1},
        }
        with cache_path.with_suffix(".meta").open("wb") as f:
            pickle.dump(meta, f, protocol=C.dump_protocol_version)
        cache_path.with_suffix(".meta").chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        # write index file
        im = DiskDatasetCache.IndexManager(cache_path)
        index_data = im.build_index_from_data(features)
        im.update(index_data)

        # rename the file after the cache has been generated
        # this doesn't work well on windows, but our server won't use windows
        # temporarily
        cache_path.with_suffix(".data").rename(cache_path)
        # the fields of the cached features are converted to the original fields
        return features.swaplevel("datetime", "instrument")

    def update(self, cache_uri, freq: str = "day"):
        cp_cache_uri = self.get_cache_dir(freq).joinpath(cache_uri)
        meta_path = cp_cache_uri.with_suffix(".meta")
        if not self.check_cache_exists(cp_cache_uri):
            self.logger.info(f"The cache {cp_cache_uri} has corrupted. It will be removed")
            self.clear_cache(cp_cache_uri)
            return 2

        im = DiskDatasetCache.IndexManager(cp_cache_uri)
        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri())}:dataset-{cache_uri}"):
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            instruments = d["info"]["instruments"]
            fields = d["info"]["fields"]
            freq = d["info"]["freq"]
            last_update_time = d["info"]["last_update"]
            inst_processors = d["info"].get("inst_processors", [])
            index_data = im.get_index()

            self.logger.debug("Updating dataset: {}".format(d))
            from .data import Inst  # pylint: disable=C0415

            if Inst.get_inst_type(instruments) == Inst.DICT:
                self.logger.info(f"The file {cache_uri} has dict cache. Skip updating")
                return 1

            # get newest calendar
            from .data import Cal  # pylint: disable=C0415

            whole_calendar = Cal.calendar(start_time=None, end_time=None, freq=freq)
            # The calendar since last updated
            new_calendar = Cal.calendar(start_time=last_update_time, end_time=None, freq=freq)

            # get append data
            if len(new_calendar) <= 1:
                # Including last updated calendar, we only get 1 item.
                # No future updating is needed.
                return 1
            else:
                # get the data needed after the historical data are removed.
                # The start index of new data
                current_index = len(whole_calendar) - len(new_calendar) + 1

                # To avoid recursive import
                from .data import ExpressionD  # pylint: disable=C0415

                # The existing data length
                lft_etd = rght_etd = 0
                for field in fields:
                    expr = ExpressionD.get_expression_instance(field)
                    l, r = expr.get_extended_window_size()
                    lft_etd = max(lft_etd, l)
                    rght_etd = max(rght_etd, r)
                # remove the period that should be updated.
                if index_data.empty:
                    # We don't have any data for such dataset. Nothing to remove
                    rm_n_period = rm_lines = 0
                else:
                    rm_n_period = min(rght_etd, index_data.shape[0])
                    rm_lines = (
                        (index_data["end"] - index_data["start"])
                        .loc[whole_calendar[current_index - rm_n_period] :]
                        .sum()
                        .item()
                    )

                data = self.provider.dataset(
                    instruments,
                    fields,
                    whole_calendar[current_index - rm_n_period],
                    new_calendar[-1],
                    freq,
                    inst_processors=inst_processors,
                )

                if not data.empty:
                    data.reset_index(inplace=True)
                    data.set_index(["datetime", "instrument"], inplace=True)
                    data.sort_index(inplace=True)
                else:
                    return 0  # No data to update cache

                store = pd.HDFStore(cp_cache_uri)
                # FIXME:
                # Because the feature cache are stored as .bin file.
                # So the series read from features are all float32.
                # However, the first dataset cache is calculated based on the
                # raw data. So the data type may be float64.
                # Different data type will result in failure of appending data
                if "/{}".format(DatasetCache.HDF_KEY) in store.keys():
                    schema = store.select(DatasetCache.HDF_KEY, start=0, stop=0)
                    for col, dtype in schema.dtypes.items():
                        data[col] = data[col].astype(dtype)
                if rm_lines > 0:
                    store.remove(key=im.KEY, start=-rm_lines)
                store.append(DatasetCache.HDF_KEY, data)
                store.close()

                # update index file
                new_index_data = im.build_index_from_data(
                    data.loc(axis=0)[whole_calendar[current_index] :, :],
                    start_index=0 if index_data.empty else index_data["end"].iloc[-1],
                )
                im.append_index(new_index_data)

                # update meta file
                d["info"]["last_update"] = str(new_calendar[-1])
                with meta_path.open("wb") as f:
                    pickle.dump(d, f, protocol=C.dump_protocol_version)
                return 0


class SimpleDatasetCache(DatasetCache):
    """Simple dataset cache that can be used locally or on client."""

    def __init__(self, provider):
        super(SimpleDatasetCache, self).__init__(provider)
        try:
            self.local_cache_path: Path = Path(C["local_cache_path"]).expanduser().resolve()
        except (KeyError, TypeError):
            self.logger.error("Assign a local_cache_path in config if you want to use this cache mechanism")
            raise
        self.logger.info(
            f"DatasetCache directory: {self.local_cache_path}, "
            f"modify the cache directory via the local_cache_path in the config"
        )

    def _uri(self, instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        instruments, fields, freq = self.normalize_uri_args(instruments, fields, freq)
        return hash_args(
            instruments, fields, start_time, end_time, freq, disk_cache, str(self.local_cache_path), inst_processors
        )

    def _dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        if disk_cache == 0:
            # In this case, data_set cache is configured but will not be used.
            return self.provider.dataset(instruments, fields, start_time, end_time, freq)
        self.local_cache_path.mkdir(exist_ok=True, parents=True)
        cache_file = self.local_cache_path.joinpath(
            self._uri(
                instruments, fields, start_time, end_time, freq, disk_cache=disk_cache, inst_processors=inst_processors
            )
        )
        gen_flag = False

        if cache_file.exists():
            if disk_cache == 1:
                # use cache
                df = pd.read_pickle(cache_file)
                return self.cache_to_origin_data(df, fields)
            elif disk_cache == 2:
                # replace cache
                gen_flag = True
        else:
            gen_flag = True

        if gen_flag:
            data = self.provider.dataset(
                instruments, normalize_cache_fields(fields), start_time, end_time, freq, inst_processors=inst_processors
            )
            data.to_pickle(cache_file)
            return self.cache_to_origin_data(data, fields)


class DatasetURICache(DatasetCache):
    """Prepared cache mechanism for server."""

    def _uri(self, instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        return hash_args(*self.normalize_uri_args(instruments, fields, freq), disk_cache, inst_processors)

    def dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]
    ):
        if "local" in C.dataset_provider.lower():
            # use LocalDatasetProvider
            return self.provider.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
            )

        if disk_cache == 0:
            # do not use data_set cache, load data from remote expression cache directly
            return self.provider.dataset(
                instruments,
                fields,
                start_time,
                end_time,
                freq,
                disk_cache,
                return_uri=False,
                inst_processors=inst_processors,
            )
        # FIXME: The cache after resample, when read again and intercepted with end_time, results in incomplete data date
        if inst_processors:
            raise ValueError(
                f"{self.__class__.__name__} does not support inst_processor. "
                f"Please use `D.features(disk_cache=0)` or `qlib.init(dataset_cache=None)`"
            )
        # use ClientDatasetProvider
        feature_uri = self._uri(
            instruments, fields, None, None, freq, disk_cache=disk_cache, inst_processors=inst_processors
        )
        value, expire = MemCacheExpire.get_cache(H["f"], feature_uri)
        mnt_feature_uri = C.dpm.get_data_uri(freq).joinpath(C.dataset_cache_dir_name).joinpath(feature_uri)
        if value is None or expire or not mnt_feature_uri.exists():
            df, uri = self.provider.dataset(
                instruments,
                fields,
                start_time,
                end_time,
                freq,
                disk_cache,
                return_uri=True,
                inst_processors=inst_processors,
            )
            # cache uri
            MemCacheExpire.set_cache(H["f"], uri, uri)
            # cache DataFrame
            # HZ['f'][uri] = df.copy()
            get_module_logger("cache").debug(f"get feature from {C.dataset_provider}")
        else:
            df = DiskDatasetCache.read_data_from_cache(mnt_feature_uri, start_time, end_time, fields)
            get_module_logger("cache").debug("get feature from uri cache")

        return df


class CalendarCache(BaseProviderCache):
    pass


class MemoryCalendarCache(CalendarCache):
    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        uri = self._uri(start_time, end_time, freq, future)
        result, expire = MemCacheExpire.get_cache(H["c"], uri)
        if result is None or expire:
            result = self.provider.calendar(start_time, end_time, freq, future)
            MemCacheExpire.set_cache(H["c"], uri, result)

            get_module_logger("data").debug(f"get calendar from {C.calendar_provider}")
        else:
            get_module_logger("data").debug("get calendar from local cache")

        return result


H = MemCache()
