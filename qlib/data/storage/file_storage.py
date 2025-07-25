# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List

import numpy as np
import pandas as pd

from qlib.utils.time import Freq
from qlib.utils.resam import resam_calendar
from qlib.config import C
from qlib.data.cache import H
from qlib.log import get_module_logger
from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT

logger = get_module_logger("file_storage")


class FileStorageMixin:
    """文件存储混合类，适用于FileXXXStorage
    子类需要具有provider_uri、freq、storage_name、file_name属性
    """

    # """注意：provider_uri的优先级：
    #   1. self._provider_uri : 如果提供了provider_uri
    #   2. qlib.config.C中的provider_uri
    
    @property
    def provider_uri(self):
        return C["provider_uri"] if getattr(self, "_provider_uri", None) is None else self._provider_uri

    @property
    def dpm(self):
        return (
            C.dpm
            if getattr(self, "_provider_uri", None) is None
            else C.DataPathManager(self._provider_uri, C.mount_path)
        )

    @property
    def support_freq(self) -> List[str]:
        _v = "_support_freq"
        if hasattr(self, _v):
            return getattr(self, _v)
        if len(self.provider_uri) == 1 and C.DEFAULT_FREQ in self.provider_uri:
            freq_l = filter(
                lambda _freq: not _freq.endswith("_future"),
                map(lambda x: x.stem, self.dpm.get_data_uri(C.DEFAULT_FREQ).joinpath("calendars").glob("*.txt")),
            )
        else:
            freq_l = self.provider_uri.keys()
        freq_l = [Freq(freq) for freq in freq_l]
        setattr(self, _v, freq_l)
        return freq_l

    @property
    def uri(self) -> Path:
        if self.freq not in self.support_freq:
            raise ValueError(f"{self.storage_name}: {self.provider_uri} does not contain data for {self.freq}")
        return self.dpm.get_data_uri(self.freq).joinpath(f"{self.storage_name}s", self.file_name)

    def check(self):
        """检查self.uri

        Raises
        -------
        ValueError
        """
        if not self.uri.exists():
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")


class FileCalendarStorage(FileStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, provider_uri: dict = None, **kwargs):
        super(FileCalendarStorage, self).__init__(freq, future, **kwargs)
        self.future = future
        # 如果提供了provider_uri，则使用格式化后的路径；否则使用默认值
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        self.enable_read_cache = True  # TODO: make it configurable
        self.region = C["region"]

    @property
    def file_name(self) -> str:
        return f"{self._freq_file}_future.txt" if self.future else f"{self._freq_file}.txt".lower()

    @property
    def _freq_file(self) -> str:
        """从文件中读取的频率"""
        if not hasattr(self, "_freq_file_cache"):
            freq = Freq(self.freq)
            if freq not in self.support_freq:
                # NOTE: uri
                #   1. If `uri` does not exist
                #       - Get the `min_uri` of the closest `freq` under the same "directory" as the `uri`
                #       - Read data from `min_uri` and resample to `freq`

                freq = Freq.get_recent_freq(freq, self.support_freq)
                if freq is None:
                    raise ValueError(f"can't find a freq from {self.support_freq} that can resample to {self.freq}!")
            self._freq_file_cache = freq
        return self._freq_file_cache

    def _read_calendar(self) -> List[CalVT]:
        # 注意：
        # 如果想加速部分日历数据的读取
        # 可以在接口中添加参数如`skip_rows: int = 0, n_rows: int = None`
        # 目前基于txt的日历不支持此功能

        if not self.uri.exists():
            self._write_calendar(values=[])

        with self.uri.open("r") as fp:
            res = []
            for line in fp.readlines():
                line = line.strip()
                if len(line) > 0:
                    res.append(line)
            return res

    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        with self.uri.open(mode=mode) as fp:
            np.savetxt(fp, values, fmt="%s", encoding="utf-8")

    @property
    def uri(self) -> Path:
        return self.dpm.get_data_uri(self._freq_file).joinpath(f"{self.storage_name}s", self.file_name)

    @property
    def data(self) -> List[CalVT]:
        # 检查文件是否存在
        self.check()
        # If cache is enabled, then return cache directly
        if self.enable_read_cache:
            key = "orig_file" + str(self.uri)
            if key not in H["c"]:
                H["c"][key] = self._read_calendar()
            _calendar = H["c"][key]
        else:
            _calendar = self._read_calendar()
        if Freq(self._freq_file) != Freq(self.freq):
            _calendar = resam_calendar(
                np.array(list(map(pd.Timestamp, _calendar))), self._freq_file, self.freq, self.region
            )
        return _calendar

    def _get_storage_freq(self) -> List[str]:
        return sorted(set(map(lambda x: x.stem.split("_")[0], self.uri.parent.glob("*.txt"))))

    def extend(self, values: Iterable[CalVT]) -> None:
        self._write_calendar(values, mode="ab")

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        self.check()
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        calendar = self._read_calendar()
        calendar = np.insert(calendar, index, value)
        self._write_calendar(values=calendar)

    def remove(self, value: CalVT) -> None:
        self.check()
        index = self.index(value)
        calendar = self._read_calendar()
        calendar = np.delete(calendar, index)
        self._write_calendar(values=calendar)

    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        calendar = self._read_calendar()
        calendar[i] = values
        self._write_calendar(values=calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        self.check()
        calendar = self._read_calendar()
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        self.check()
        return self._read_calendar()[i]

    def __len__(self) -> int:
        return len(self.data)


class FileInstrumentStorage(FileStorageMixin, InstrumentStorage):
    INSTRUMENT_SEP = "\t"
    INSTRUMENT_START_FIELD = "start_datetime"  # 开始时间字段
    INSTRUMENT_END_FIELD = "end_datetime"    # 结束时间字段
    SYMBOL_FIELD_NAME = "instrument"         # 工具代码字段

    def __init__(self, market: str, freq: str, provider_uri: dict = None, **kwargs):
        super(FileInstrumentStorage, self).__init__(market, freq, **kwargs)
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        # 文件名格式化为小写市场名称
        self.file_name = f"{market.lower()}.txt"

    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        # 如果文件不存在，则创建空文件
        if not self.uri.exists():
            self._write_instrument()

        _instruments = dict()
        df = pd.read_csv(
            self.uri,
            sep="\t",
            usecols=[0, 1, 2],
            names=[self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD],
            dtype={self.SYMBOL_FIELD_NAME: str},
            parse_dates=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD],
        )
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))
        return _instruments

    def _write_instrument(self, data: Dict[InstKT, InstVT] = None) -> None:
        if not data:
            with self.uri.open("w") as _:
                pass
            return

        res = []
        # 将工具数据转换为DataFrame格式
        for inst, v_list in data.items():
            _df = pd.DataFrame(v_list, columns=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD])
            _df[self.SYMBOL_FIELD_NAME] = inst
            res.append(_df)

        df = pd.concat(res, sort=False)
        # 保存工具数据到文件
        df.loc[:, [self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD]].to_csv(
            self.uri, header=False, sep=self.INSTRUMENT_SEP, index=False
        )
        df.to_csv(self.uri, sep="\t", encoding="utf-8", header=False, index=False)

    def clear(self) -> None:
        # 清空工具数据
        self._write_instrument(data={})

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        self.check()
        return self._read_instrument()

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        inst = self._read_instrument()
        inst[k] = v
        self._write_instrument(inst)

    def __delitem__(self, k: InstKT) -> None:
        self.check()
        inst = self._read_instrument()
        del inst[k]
        self._write_instrument(inst)

    def __getitem__(self, k: InstKT) -> InstVT:
        self.check()
        return self._read_instrument()[k]

    def update(self, *args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
        inst = self._read_instrument()
        if args:
            other = args[0]  # type: dict
            if isinstance(other, Mapping):
                for key in other:
                    inst[key] = other[key]
            elif hasattr(other, "keys"):
                for key in other.keys():
                    inst[key] = other[key]
            else:
                for key, value in other:
                    inst[key] = value
        for key, value in kwargs.items():
            inst[key] = value

        self._write_instrument(inst)

    def __len__(self) -> int:
        return len(self.data)


class FileFeatureStorage(FileStorageMixin, FeatureStorage):
    def __init__(self, instrument: str, field: str, freq: str, provider_uri: dict = None, **kwargs):
        super(FileFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        # 文件名格式化为小写工具/字段/频率.bin
        self.file_name = f"{instrument.lower()}/{field.lower()}.{freq.lower()}.bin"

    def clear(self):
        with self.uri.open("wb") as _:
            pass

    @property
    def data(self) -> pd.Series:
        return self[:]

    def write(self, data_array: Union[List, np.ndarray], index: int = None) -> None:
        if len(data_array) == 0:
            logger.info(
                "数据数组长度为0，写入操作已取消。"
                "如果需要清空特征存储，请执行: FeatureStorage.clear"
            )
            return
        if not self.uri.exists():
            # 文件不存在，直接写入数据
            index = 0 if index is None else index
            with self.uri.open("wb") as fp:
                np.hstack([index, data_array]).astype("<f").tofile(fp)
        else:
            if index is None or index > self.end_index:
                # 追加数据到文件末尾
                index = 0 if index is None else index
                with self.uri.open("ab+") as fp:
                    # 填充缺失索引位置为NaN
                    np.hstack([[np.nan] * (index - self.end_index - 1), data_array]).astype("<f").tofile(fp)
            else:
                # 重写指定位置的数据
                with self.uri.open("rb+") as fp:
                    _old_data = np.fromfile(fp, dtype="<f")
                    _old_index = _old_data[0]
                    _old_df = pd.DataFrame(
                        _old_data[1:], index=range(_old_index, _old_index + len(_old_data) - 1), columns=["old"]
                    )
                    fp.seek(0)
                    _new_df = pd.DataFrame(data_array, index=range(index, index + len(data_array)), columns=["new"])
                    _df = pd.concat([_old_df, _new_df], sort=False, axis=1)
                    _df = _df.reindex(range(_df.index.min(), _df.index.max() + 1))
                    # 新数据优先，缺失值用旧数据填充
                    _df["new"].fillna(_df["old"]).values.astype("<f").tofile(fp)

    @property
    def start_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        with self.uri.open("rb") as fp:
            index = int(np.frombuffer(fp.read(4), dtype="<f")[0])
        return index

    @property
    def end_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        # 下一个数据追加的索引位置为 `end_index + 1`
        return self.start_index + len(self) - 1

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:
        if not self.uri.exists():
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return pd.Series(dtype=np.float32)
            else:
                raise TypeError(f"type(i) = {type(i)}")

        storage_start_index = self.start_index
        storage_end_index = self.end_index
        with self.uri.open("rb") as fp:
            if isinstance(i, int):
                # 检查索引是否在有效范围内
                if storage_start_index > i:
                    raise IndexError(f"索引{i}超出范围，起始索引为{storage_start_index}")
                # 定位到指定索引位置
                fp.seek(4 * (i - storage_start_index) + 4)
                return i, struct.unpack("f", fp.read(4))[0]
            elif isinstance(i, slice):
                # 处理切片索引
                start_index = storage_start_index if i.start is None else i.start
                end_index = storage_end_index if i.stop is None else i.stop - 1
                si = max(start_index, storage_start_index)
                if si > end_index:
                    return pd.Series(dtype=np.float32)
                # 定位到起始位置
                fp.seek(4 * (si - storage_start_index) + 4)
                # 读取指定长度的数据
                count = end_index - si + 1
                data = np.frombuffer(fp.read(4 * count), dtype="<f")
                return pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            else:
                raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        self.check()
        return self.uri.stat().st_size // 4 - 1
