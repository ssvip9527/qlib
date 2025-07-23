# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH


device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


def _create_ts_slices(index, seq_len):
    """从pandas索引创建时间序列切片

    参数:
        index (pd.MultiIndex): 按<instrument, datetime>顺序排列的pandas多重索引
        seq_len (int): 序列长度
    """
    assert index.is_lexsorted(), "index should be sorted"

    # number of dates for each code
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0, group_keys=False).size().values

    # start_index for each code
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # all the [start, stop) indices of features
    # features btw [start, stop) are used to predict the `stop - 1` label
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices)

    return slices


def _get_date_parse_fn(target):
    """获取日期解析函数

    此方法用于将日期参数解析为目标类型。

    示例:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, pd.Timestamp):
        _fn = lambda x: pd.Timestamp(x)  # Timestamp('2020-01-01')
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    elif isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    else:
        _fn = lambda x: x
    return _fn


class MTSDatasetH(DatasetH):
    """内存增强时间序列数据集

    参数:
        handler (DataHandler): 数据处理器
        segments (dict): 数据分割段
        seq_len (int): 时间序列序列长度
        horizon (int): 标签视野（用于屏蔽TRA的历史损失）
        num_states (int): 要添加的内存状态数量（用于TRA）
        batch_size (int): 批次大小（<0表示每日批次）
        shuffle (bool): 是否打乱数据
        pin_memory (bool): 是否将数据固定到GPU内存
        drop_last (bool): 是否丢弃最后一个小于batch_size的批次
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=60,
        horizon=0,
        num_states=1,
        batch_size=-1,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        **kwargs,
    ):
        assert horizon > 0, "please specify `horizon` to avoid data leakage"

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.params = (batch_size, drop_last, shuffle)  # for train/eval switch

        super().__init__(handler, segments, **kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        super().setup_data()

        # change index to <code, date>
        # NOTE: we will use inplace sort to reduce memory use
        df = self.handler._data
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        self._data = df["feature"].values.astype("float32")
        self._label = df["label"].squeeze().astype("float32")
        self._index = df.index

        # add memory to feature
        self._data = np.c_[self._data, np.zeros((len(self._data), self.num_states), dtype=np.float32)]

        # padding tensor
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)

        # pin memory
        if self.pin_memory:
            self._data = _to_tensor(self._data)
            self._label = _to_tensor(self._label)
            self.zeros = _to_tensor(self.zeros)

        # create batch slices
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)

        # create daily slices
        index = [slc.stop - 1 for slc in self.batch_slices]
        act_index = self.restore_index(index)
        daily_slices = {date: [] for date in sorted(act_index.unique(level=1))}
        for i, (code, date) in enumerate(act_index):
            daily_slices[date].append(self.batch_slices[i])
        self.daily_slices = list(daily_slices.values())

    def _prepare_seg(self, slc, **kwargs):
        fn = _get_date_parse_fn(self._index[0][1])

        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError(f"This type of input is not supported")
        start_date = fn(start)
        end_date = fn(stop)
        obj = copy.copy(self)  # shallow copy
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data
        obj._label = self._label
        obj._index = self._index
        new_batch_slices = []
        for batch_slc in self.batch_slices:
            date = self._index[batch_slc.stop - 1][1]
            if start_date <= date <= end_date:
                new_batch_slices.append(batch_slc)
        obj.batch_slices = np.array(new_batch_slices)
        new_daily_slices = []
        for daily_slc in self.daily_slices:
            date = self._index[daily_slc[0].stop - 1][1]
            if start_date <= date <= end_date:
                new_daily_slices.append(daily_slc)
        obj.daily_slices = new_daily_slices
        return obj

    def restore_index(self, index):
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return self._index[index]

    def assign_data(self, index, vals):
        if isinstance(self._data, torch.Tensor):
            vals = _to_tensor(vals)
        elif isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
        self._data[index, -self.num_states :] = vals

    def clear_memory(self):
        self._data[:, -self.num_states :] = 0

    # TODO: better train/eval mode design
    def train(self):
        """启用训练模式"""
        self.batch_size, self.drop_last, self.shuffle = self.params

    def eval(self):
        """启用评估模式"""
        self.batch_size = -1
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:
            slices = self.daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:
            slices = self.batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)

        for i in range(len(slices))[::batch_size]:
            if self.drop_last and i + batch_size > len(slices):
                break
            # get slices for this batch
            slices_subset = slices[i : i + batch_size]
            if self.batch_size < 0:
                slices_subset = np.concatenate(slices_subset)
            # collect data
            data = []
            label = []
            index = []
            for slc in slices_subset:
                _data = self._data[slc].clone() if self.pin_memory else self._data[slc].copy()
                if len(_data) != self.seq_len:
                    if self.pin_memory:
                        _data = torch.cat([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                    else:
                        _data = np.concatenate([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                if self.num_states > 0:
                    _data[-self.horizon :, -self.num_states :] = 0
                data.append(_data)
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)
            # concate
            index = torch.tensor(index, device=device)
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
                label = torch.stack(label)
            else:
                data = _to_tensor(np.stack(data))
                label = _to_tensor(np.stack(label))
            # yield -> generator
            yield {"data": data, "label": label, "index": index}
