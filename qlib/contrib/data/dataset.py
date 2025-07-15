# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import warnings
import numpy as np
import pandas as pd
from qlib.utils.data import guess_horizon
from qlib.utils import init_instance_by_config

from qlib.data.dataset import DatasetH


device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)  # pylint: disable=E1101
    return x


def _create_ts_slices(index, seq_len):
    """
    从pandas索引创建时间序列切片

    参数:
        index (pd.MultiIndex): pandas多级索引，顺序为<instrument, datetime>
        seq_len (int): 序列长度
    """
    assert isinstance(index, pd.MultiIndex), "unsupported index type"
    assert seq_len > 0, "sequence length should be larger than 0"
    assert index.is_monotonic_increasing, "index should be sorted"

    # 每个instrument的日期数量
    sample_count_by_insts = index.to_series().groupby(level=0, group_keys=False).size().values

    # 每个instrument的起始索引
    start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
    start_index_of_insts[0] = 0

    # 所有特征的[start, stop)索引
    # [start, stop)之间的特征将用于预测`stop - 1`处的标签
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices, dtype="object")

    assert len(slices) == len(index)  # the i-th slice = index[i]

    return slices


def _get_date_parse_fn(target):
    """获取日期解析函数

    此方法用于将日期参数解析为目标类型。

    示例：
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, int):

        def _fn(x):
            return int(str(x).replace("-", "")[:8])  # 20200201

    elif isinstance(target, str) and len(target) == 8:

        def _fn(x):
            return str(x).replace("-", "")[:8]  # '20200201'

    else:

        def _fn(x):
            return x  # '2021-01-01'

    return _fn


def _maybe_padding(x, seq_len, zeros=None):
    """用零填充二维<时间*特征>数据

    参数：
        x (np.ndarray): 二维数据，形状为<时间*特征>
        seq_len (int): 目标序列长度
        zeros (np.ndarray): 零矩阵，形状为<seq_len * feature>
    """
    assert seq_len > 0, "sequence length should be larger than 0"
    if zeros is None:
        zeros = np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    else:
        assert len(zeros) >= seq_len, "zeros matrix is not large enough for padding"
    if len(x) != seq_len:  # padding zeros
        x = np.concatenate([zeros[: seq_len - len(x), : x.shape[1]], x], axis=0)
    return x


class MTSDatasetH(DatasetH):
    """内存增强时间序列数据集

    参数:
        handler (DataHandler): 数据处理器
        segments (dict): 数据分割片段
        seq_len (int): 时间序列长度
        horizon (int): 标签预测周期
        num_states (int): 要添加的内存状态数量
        memory_mode (str): 内存模式（daily或sample）
        batch_size (int): 批次大小（<0将使用按日采样）
        n_samples (int): 同一天内的样本数量
        shuffle (bool): 是否打乱数据
        drop_last (bool): 是否丢弃最后一个小于batch_size的批次
        input_size (int): 将展平的行重塑为此输入大小（向后兼容）
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=60,
        horizon=0,
        num_states=0,
        memory_mode="sample",
        batch_size=-1,
        n_samples=None,
        shuffle=True,
        drop_last=False,
        input_size=None,
        **kwargs,
    ):
        if horizon == 0:
            # 尝试猜测预测周期
            if isinstance(handler, (dict, str)):
                handler = init_instance_by_config(handler)
            assert "label" in getattr(handler.data_loader, "fields", None)
            label = handler.data_loader.fields["label"][0][0]
            horizon = guess_horizon([label])

        assert num_states == 0 or horizon > 0, "请指定`horizon`以避免数据泄露"
        assert memory_mode in ["sample", "daily"], "不支持的内存模式"
        assert memory_mode == "sample" or batch_size < 0, "daily内存模式需要按日采样（`batch_size < 0`）"
        assert batch_size != 0, "无效的批次大小"

        if batch_size > 0 and n_samples is not None:
            warnings.warn("`n_samples`仅可用于按日采样（`batch_size < 0`）")

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.memory_mode = memory_mode
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.input_size = input_size
        self.params = (batch_size, n_samples, drop_last, shuffle)  # 用于`train/eval`切换

        super().__init__(handler, segments, **kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        super().setup_data(**kwargs)

        if handler_kwargs is not None:
            self.handler.setup_data(**handler_kwargs)

        # 预获取数据并将索引更改为<code, date>
        # 注意：我们将使用原地排序以减少内存使用
        try:
            df = self.handler._learn.copy()  # use copy otherwise recorder will fail
            # FIXME: 当前不支持从`_learn`切换到`_infer`进行推理
        except Exception:
            warnings.warn("无法访问`_learn`，将加载原始数据")
            df = self.handler._data.copy()
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        # 转换为numpy数组
        self._data = df["feature"].values.astype("float32")
        np.nan_to_num(self._data, copy=False)  # 注意：填充NaN以防用户忘记使用fillna处理器
        self._label = df["label"].squeeze().values.astype("float32")
        self._index = df.index

        if self.input_size is not None and self.input_size != self._data.shape[1]:
            warnings.warn("数据形状与input_size不同，数据将被重塑")
            assert self._data.shape[1] % self.input_size == 0, "数据不匹配，请检查`input_size`"

        # 创建批次切片
        self._batch_slices = _create_ts_slices(self._index, self.seq_len)

        # 创建每日切片
        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}  # 按日期排序
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append(self._batch_slices[i])
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(list(daily_slices.keys()))  # 索引为原始日期索引

        # 添加内存（按样本和按日）
        if self.memory_mode == "sample":
            self._memory = np.zeros((len(self._data), self.num_states), dtype=np.float32)
        elif self.memory_mode == "daily":
            self._memory = np.zeros((len(self._daily_index), self.num_states), dtype=np.float32)
        else:
            raise ValueError(f"invalid memory_mode `{self.memory_mode}`")

        # 填充张量
        self._zeros = np.zeros((self.seq_len, max(self.num_states, self._data.shape[1])), dtype=np.float32)

    def _prepare_seg(self, slc, **kwargs):
        fn = _get_date_parse_fn(self._index[0][1])
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError(f"This type of input is not supported")
        start_date = pd.Timestamp(fn(start))
        end_date = pd.Timestamp(fn(stop))
        obj = copy.copy(self)  # shallow copy
        # 注意：Seriable会禁用`self._data`的复制，因此我们在此手动赋值
        obj._data = self._data  # reference (no copy)
        obj._label = self._label
        obj._index = self._index
        obj._memory = self._memory
        obj._zeros = self._zeros
        # update index for this batch
        date_index = self._index.get_level_values(1)
        obj._batch_slices = self._batch_slices[(date_index >= start_date) & (date_index <= end_date)]
        mask = (self._daily_index.values >= start_date) & (self._daily_index.values <= end_date)
        obj._daily_slices = self._daily_slices[mask]
        obj._daily_index = self._daily_index[mask]
        return obj

    def restore_index(self, index):
        return self._index[index]

    def restore_daily_index(self, daily_index):
        return pd.Index(self._daily_index.loc[daily_index])

    def assign_data(self, index, vals):
        if self.num_states == 0:
            raise ValueError("cannot assign data as `num_states==0`")
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        self._memory[index] = vals

    def clear_memory(self):
        if self.num_states == 0:
            raise ValueError("cannot clear memory as `num_states==0`")
        self._memory[:] = 0

    def train(self):
        """启用训练模式"""
        self.batch_size, self.n_samples, self.drop_last, self.shuffle = self.params

    def eval(self):
        """启用评估模式"""
        self.batch_size = -1
        self.n_samples = None
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:  # daily sampling
            slices = self._daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:  # normal sampling
            slices = self._batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        indices = np.arange(len(slices))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(indices))[::batch_size]:
            if self.drop_last and i + batch_size > len(indices):
                break

            data = []  # store features
            label = []  # store labels
            index = []  # store index
            state = []  # store memory states
            daily_index = []  # store daily index
            daily_count = []  # store number of samples for each day

            for j in indices[i : i + batch_size]:
                # 常规采样: self.batch_size > 0 => slices is a list => slices_subset is a slice
                # 按日采样: self.batch_size < 0 => slices is a nested list => slices_subset is a list
                slices_subset = slices[j]

                # daily sampling
                # each slices_subset contains a list of slices for multiple stocks
                # NOTE: daily sampling is used in 1) eval mode, 2) train mode with self.batch_size < 0
                if self.batch_size < 0:
                    # 存储每日索引
                    idx = self._daily_index.index[j]  # daily_index.index is the index of the original data
                    daily_index.append(idx)

                    # 如指定则存储每日内存
                    # 注意：每日内存始终需要按日采样（self.batch_size < 0）
                    if self.memory_mode == "daily":
                        slc = slice(max(idx - self.seq_len - self.horizon, 0), max(idx - self.horizon, 0))
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros))

                    # 对股票进行下采样并存储数量
                    if self.n_samples and 0 < self.n_samples < len(slices_subset):  # 日内子采样
                        slices_subset = np.random.choice(slices_subset, self.n_samples, replace=False)
                    daily_count.append(len(slices_subset))

                # 常规采样
                # each slices_subset is a single slice
                # 注意：常规采样用于self.batch_size > 0的训练模式
                else:
                    slices_subset = [slices_subset]

                for slc in slices_subset:
                    # 通过`input_size`向后兼容Alpha360数据
                    if self.input_size:
                        data.append(self._data[slc.stop - 1].reshape(self.input_size, -1).T)
                    else:
                        data.append(_maybe_padding(self._data[slc], self.seq_len, self._zeros))

                    if self.memory_mode == "sample":
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros)[: -self.horizon])

                    label.append(self._label[slc.stop - 1])
                    index.append(slc.stop - 1)

                    # 结束切片循环

                # 结束索引批次循环

            # 拼接
            data = _to_tensor(np.stack(data))
            state = _to_tensor(np.stack(state))
            label = _to_tensor(np.stack(label))
            index = np.array(index)
            daily_index = np.array(daily_index)
            daily_count = np.array(daily_count)

            # yield -> 生成器
            yield {
                "data": data,
                "label": label,
                "state": state,
                "index": index,
                "daily_index": daily_index,
                "daily_count": daily_count,
            }

        # 结束索引循环
