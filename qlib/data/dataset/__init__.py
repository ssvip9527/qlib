from ...utils.serial import Serializable
from typing import Callable, Union, List, Tuple, Dict, Text, Optional
from ...utils import init_instance_by_config, np_ffill, time_to_slc_point
from ...log import get_module_logger
from .handler import DataHandler, DataHandlerLP
from copy import copy, deepcopy
from inspect import getfullargspec
import pandas as pd
import numpy as np
import bisect
from ...utils import lazy_sort_index
from .utils import get_level_index


class Dataset(Serializable):
    """
    为模型训练和推理准备数据。
    """

    def __init__(self, **kwargs):
        """
        初始化旨在完成以下步骤：

        - 初始化子实例和数据集的状态（准备数据所需的信息）
            - 用于准备数据的基本状态名称不应以'_'开头，以便在序列化时可以保存到磁盘。

        - 设置数据
            - 数据相关属性的名称应以'_'开头，以便在序列化时不会保存到磁盘。

        数据可以指定计算准备所需基本数据的信息
        """
        self.setup_data(**kwargs)
        super().__init__()

    def config(self, **kwargs):
        """
        config is designed to configure and parameters that cannot be learned from the data
        """
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        """
        Setup the data.

        We split the setup_data function for following situation:

        - User have a Dataset object with learned status on disk.

        - User load the Dataset object from the disk.

        - User call `setup_data` to load new data.

        - User prepare data for model based on previous status.
        """

    def prepare(self, **kwargs) -> object:
        """
        数据集的类型取决于模型（可以是pd.DataFrame、pytorch.DataLoader等）。
        参数应指定准备数据的范围。
        该方法应：
        - 处理数据
        - 返回处理后的数据

        返回
        -------
        object:
            返回处理后的对象
        """


class DatasetH(Dataset):
    """
    带有数据处理器(DataHandler)的数据集。

    用户应尝试将数据预处理函数放入处理器中。
    只有以下数据处理函数应放在数据集中：
    - 与特定模型相关的处理
    - 与数据拆分相关的处理
    """

    def __init__(
        self,
        handler: Union[Dict, DataHandler],
        segments: Dict[Text, Tuple],
        fetch_kwargs: Dict = {},
        **kwargs,
    ):
        """
        Setup the underlying data.

        :param handler: 处理器可以是：
            - `DataHandler`的实例
            - `DataHandler`的配置。请参考`DataHandler`
        :type handler: Union[dict, DataHandler]
        :param segments: 描述数据分段的选项。以下是一些示例：
            .. code-block::

                1) 'segments': {
                        'train': ("2008-01-01", "2014-12-31"),
                        'valid': ("2017-01-01", "2020-08-01",),
                        'test': ("2015-01-01", "2016-12-31",),
                    }
                2) 'segments': {
                        'insample': ("2008-01-01", "2014-12-31"),
                        'outsample': ("2017-01-01", "2020-08-01",),
                    }
        :type segments: dict
        """
        self.handler: DataHandler = init_instance_by_config(handler, accept_types=DataHandler)
        self.segments = segments.copy()
        self.fetch_kwargs = copy(fetch_kwargs)
        super().__init__(**kwargs)

    def config(self, handler_kwargs: dict = None, **kwargs):
        """
        Initialize the DatasetH

        Parameters
        ----------
        handler_kwargs : dict
            Config of DataHandler, which could include the following arguments:

            - arguments of DataHandler.conf_data, such as 'instruments', 'start_time' and 'end_time'.

        kwargs : dict
            Config of DatasetH, such as

            - segments : dict
                Config of segments which is same as 'segments' in self.__init__

        """
        if handler_kwargs is not None:
            self.handler.config(**handler_kwargs)
        if "segments" in kwargs:
            self.segments = deepcopy(kwargs.pop("segments"))
        super().config(**kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        """
        Setup the Data

        Parameters
        ----------
        handler_kwargs : dict
            init arguments of DataHandler, which could include the following arguments:

            - init_type : Init Type of Handler

            - enable_cache : whether to enable cache

        """
        super().setup_data(**kwargs)
        if handler_kwargs is not None:
            self.handler.setup_data(**handler_kwargs)

    def __repr__(self):
        return "{name}(handler={handler}, segments={segments})".format(
            name=self.__class__.__name__, handler=self.handler, segments=self.segments
        )

    def _prepare_seg(self, slc, **kwargs):
        """
        Give a query, retrieve the according data

        Parameters
        ----------
        slc : please refer to the docs of `prepare`
                NOTE: it may not be an instance of slice. It may be a segment of `segments` from `def prepare`
        """
        if hasattr(self, "fetch_kwargs"):
            return self.handler.fetch(slc, **kwargs, **self.fetch_kwargs)
        else:
            return self.handler.fetch(slc, **kwargs)

    def prepare(
        self,
        segments: Union[List[Text], Tuple[Text], Text, slice, pd.Index],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        Prepare the data for learning and inference.

        Parameters
        ----------
        segments : Union[List[Text], Tuple[Text], Text, slice]
            Describe the scope of the data to be prepared
            Here are some examples:

            - 'train'

            - ['train', 'valid']

        col_set : str
            The col_set will be passed to self.handler when fetching data.
            TODO: make it automatic:

            - select DK_I for test data
            - select DK_L for training data.
        data_key : str
            The data to fetch:  DK_*
            Default is DK_I, which indicate fetching data for **inference**.

        kwargs :
            The parameters that kwargs may contain:
                flt_col : str
                    It only exists in TSDatasetH, can be used to add a column of data(True or False) to filter data.
                    This parameter is only supported when it is an instance of TSDatasetH.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]:

        Raises
        ------
        NotImplementedError:
        """
        seg_kwargs = {"col_set": col_set, "data_key": data_key}
        seg_kwargs.update(kwargs)

        # Conflictions may happen here
        # - The fetched data and the segment key may both be string
        # To resolve the confliction
        # - The segment name will have higher priorities

        # 1) Use it as segment name first
        # 1.1) directly fetch split like "train" "valid" "test"
        if isinstance(segments, str) and segments in self.segments:
            return self._prepare_seg(self.segments[segments], **seg_kwargs)

        # 1.2) fetch multiple splits like ["train", "valid"] ["train", "valid", "test"]
        if isinstance(segments, (list, tuple)) and all(seg in self.segments for seg in segments):
            return [self._prepare_seg(self.segments[seg], **seg_kwargs) for seg in segments]

        # 2) Use pass it directly to prepare a single seg
        return self._prepare_seg(segments, **seg_kwargs)

    # helper functions
    @staticmethod
    def get_min_time(segments):
        return DatasetH._get_extrema(segments, 0, (lambda a, b: a > b))

    @staticmethod
    def get_max_time(segments):
        return DatasetH._get_extrema(segments, 1, (lambda a, b: a < b))

    @staticmethod
    def _get_extrema(segments, idx: int, cmp: Callable, key_func=pd.Timestamp):
        """it will act like sort and return the max value or None"""
        candidate = None
        for _, seg in segments.items():
            point = seg[idx]
            if point is None:
                # None indicates unbounded, return directly
                return None
            elif candidate is None or cmp(key_func(candidate), key_func(point)):
                candidate = point
        return candidate


class TSDataSampler:
    """
    (T)ime-(S)eries DataSampler
    This is the result of TSDatasetH

    It works like `torch.data.utils.Dataset`, it provides a very convenient interface for constructing time-series
    dataset based on tabular data.
    - On time step dimension, the smaller index indicates the historical data and the larger index indicates the future
      data.

    If user have further requirements for processing data, user could process them based on `TSDataSampler` or create
    more powerful subclasses.

    Known Issues:
    - For performance issues, this Sampler will convert dataframe into arrays for better performance. This could result
      in a different data type


    Indices design:
        TSDataSampler has a index mechanism to help users query time-series data efficiently.

        The definition of related variables:
            data_arr: np.ndarray
                The original data. it will contains all the original data.
                The querying are often for time-series of a specific stock.
                By leveraging this data charactoristics to speed up querying, the multi-index of data_arr is rearranged in (instrument, datetime) order

            data_index: pd.MultiIndex with index order <instrument, datetime>
                it has the same shape with `idx_map`. Each elements of them are expected to be aligned.

            idx_map: np.ndarray
                It is the indexable data. It originates from data_arr, and then filtered by 1) `start` and `end`  2) `flt_data`
                    The extra data in data_arr is useful in following cases
                    1) creating meaningful time series data before `start` instead of padding them with zeros
                    2) some data are excluded by `flt_data` (e.g. no <X, y> sample pair for that index). but they are still used in time-series in X

                Finnally, it will look like.

                array([[  0,   0],
                       [  1,   0],
                       [  2,   0],
                       ...,
                       [241, 348],
                       [242, 348],
                       [243, 348]], dtype=int32)

                It list all indexable data(some data only used in historical time series data may not be indexabla), the values are the corresponding row and col in idx_df
            idx_df: pd.DataFrame
                It aims to map the <datetime, instrument> key to the original position in data_arr

                For example, it may look like (NOTE: the index for a instrument time-series is continoues in memory)

                    instrument SH600000 SH600008 SH600009 SH600010 SH600011 SH600015  ...
                    datetime
                    2017-01-03        0      242      473      717      NaN      974  ...
                    2017-01-04        1      243      474      718      NaN      975  ...
                    2017-01-05        2      244      475      719      NaN      976  ...
                    2017-01-06        3      245      476      720      NaN      977  ...

            With these two indices(idx_map, idx_df) and original data(data_arr), we can make the following queries fast (implemented in __getitem__)
            (1) Get the i-th indexable sample(time-series):   (indexable sample index) -> [idx_map] -> (row col) -> [idx_df] -> (index in data_arr)
            (2) Get the specific sample by <datetime, instrument>:  (<datetime, instrument>, i.e. <row, col>) -> [idx_df] -> (index in data_arr)
            (3) Get the index of a time-series data:   (get the <row, col>, refer to (1), (2)) -> [idx_df] -> (all indices in data_arr for time-series)
    """

    # Please refer to the docstring of TSDataSampler for the definition of following attributes
    data_arr: np.ndarray
    data_index: pd.MultiIndex
    idx_map: np.ndarray
    idx_df: pd.DataFrame

    def __init__(
        self,
        data: pd.DataFrame,
        start,
        end,
        step_len: int,
        fillna_type: str = "none",
        dtype=None,
        flt_data=None,
    ):
        """
        Build a dataset which looks like torch.data.utils.Dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The raw tabular data whose index order is <"datetime", "instrument">
        start :
            The indexable start time
        end :
            The indexable end time
        step_len : int
            The length of the time-series step
        fillna_type : int
            How will qlib handle the sample if there is on sample in a specific date.
            none:
                fill with np.nan
            ffill:
                ffill with previous sample
            ffill+bfill:
                ffill with previous samples first and fill with later samples second
        flt_data : pd.Series
            a column of data(True or False) to filter data. Its index order is <"datetime", "instrument">
            This feature is essential because:
            - We want some sample not included due to label-based filtering, but we can't filter them at the beginning due to the features is still important in the feature.
            None:
                kepp all data

        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        assert get_level_index(data, "datetime") == 0
        self.data = data.swaplevel().sort_index().copy()
        data.drop(
            data.columns, axis=1, inplace=True
        )  # data is useless since it's passed to a transposed one, hard code to free the memory of this dataframe to avoid three big dataframe in the memory(including: data, self.data, self.data_arr)

        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.data_arr = np.array(**kwargs)  # Get index from numpy.array will much faster than DataFrame.values!
        # NOTE:
        # - append last line with full NaN for better performance in `__getitem__`
        # - Keep the same dtype will result in a better performance
        self.data_arr = np.append(
            self.data_arr,
            np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype),
            axis=0,
        )
        self.nan_idx = len(self.data_arr) - 1  # The last line is all NaN; setting it to -1 can cause bug #1716

        # the data type will be changed
        # The index of usable data is between start_idx and end_idx
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)

        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            # NOTE: bool(np.nan) is True !!!!!!!!
            # make sure reindex comes first. Otherwise extra NaN may appear.
            flt_data = flt_data.swaplevel()
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(bool)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)
        self.idx_map, self.data_index = self.slice_idx_map_and_data_index(
            self.idx_map, self.idx_df, self.data_index, start, end
        )

        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)  # for better performance
        del self.data  # save memory

    @staticmethod
    def slice_idx_map_and_data_index(
        idx_map,
        idx_df,
        data_index,
        start,
        end,
    ):
        assert (
            len(idx_map) == data_index.shape[0]
        )  # make sure idx_map and data_index is same so index of idx_map can be used on data_index

        start_row_idx, end_row_idx = idx_df.index.slice_locs(start=time_to_slc_point(start), end=time_to_slc_point(end))

        time_flter_idx = (idx_map[:, 0] < end_row_idx) & (idx_map[:, 0] >= start_row_idx)
        return idx_map[time_flter_idx], data_index[time_flter_idx]

    @staticmethod
    def idx_map2arr(idx_map):
        # pytorch data sampler will have better memory control without large dict or list
        # - https://github.com/pytorch/pytorch/issues/13243
        # - https://github.com/airctic/icevision/issues/613
        # So we convert the dict into int array.
        # The arr_map is expected to behave the same as idx_map

        dtype = np.int32
        # set a index out of bound to indicate the none existing
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        """
        Get the pandas index of the data, it will be useful in following scenarios
        - Special sampler will be used (e.g. user want to sample day by day)
        """
        return self.data_index.swaplevel()  # to align the order of multiple index of original data received by __init__

    def config(self, **kwargs):
        # Config the attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        The relation of the data

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame with index in order <instrument, datetime>

                                      RSQR5     RESI5     WVMA5    LABEL0
            instrument datetime
            SH600000   2017-01-03  0.016389  0.461632 -1.154788 -0.048056
                       2017-01-04  0.884545 -0.110597 -1.059332 -0.030139
                       2017-01-05  0.507540 -0.535493 -1.099665 -0.644983
                       2017-01-06 -1.267771 -0.669685 -1.636733  0.295366
                       2017-01-09  0.339346  0.074317 -0.984989  0.765540

        Returns
        -------
        Tuple[pd.DataFrame, dict]:
            1) the first element:  reshape the original index into a <datetime(row), instrument(column)> 2D dataframe
                instrument SH600000 SH600008 SH600009 SH600010 SH600011 SH600015  ...
                datetime
                2017-01-03        0      242      473      717      NaN      974  ...
                2017-01-04        1      243      474      718      NaN      975  ...
                2017-01-05        2      244      475      719      NaN      976  ...
                2017-01-06        3      245      476      720      NaN      977  ...
            2) the second element:  {<original index>: <row, col>}
        """
        # object incase of pandas converting int to float
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = lazy_sort_index(idx_df, axis=1).T

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        get series indices of self.data_arr from the row, col indices of self.idx_df

        Parameters
        ----------
        row : int
            the row in self.idx_df
        col : int
            the col in self.idx_df

        Returns
        -------
        np.array:
            The indices of data of the data
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        """
        get the col index and row index of a given sample index in self.idx_df

        Parameters
        ----------
        idx :
            the input of  `__getitem__`

        Returns
        -------
        Tuple[int]:
            the row and col index
        """
        # The the right row number `i` and col number `j` in idx_df
        if isinstance(idx, (int, np.integer)):
            real_idx = idx
            if 0 <= real_idx < len(self.idx_map):
                i, j = self.idx_map[real_idx]  # TODO: The performance of this line is not good
            else:
                raise KeyError(f"{real_idx} is out of [0, {len(self.idx_map)})")
        elif isinstance(idx, tuple):
            # <TSDataSampler object>["datetime", "instruments"]
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            # NOTE: This relies on the idx_df columns sorted in `__init__`
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        """
        # We have two method to get the time-series of a sample
        tsds is a instance of TSDataSampler

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        tsds['2016-12-31', "SZ300315"]

        # The return value will be similar to the data retrieved by following code
        df.loc(axis=0)['2015-01-01':'2016-12-31', "SZ300315"].iloc[-30:]

        Parameters
        ----------
        idx : Union[int, Tuple[object, str]]
        """
        # Multi-index type
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        # 1) for better performance, use the last nan line for padding the lost date
        # 2) In case of precision problems. We use np.float64. # TODO: I'm not sure if whether np.float64 will result in
        # precision problems. It will not cause any problems in my tests at least
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)

        if (np.diff(indices) == 1).all():  # slicing instead of indexing for speeding up.
            data = self.data_arr[indices[0] : indices[-1] + 1]
        else:
            data = self.data_arr[indices]
        if isinstance(idx, mtit):
            # if we get multiple indexes, addition dimension should be added.
            # <sample_idx, step_idx, feature_idx>
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        return len(self.idx_map)


class TSDatasetH(DatasetH):
    """
    (T)ime-(S)eries Dataset (H)andler


    Convert the tabular data to Time-Series data

    Requirements analysis

    The typical workflow of a user to get time-series data for an sample
    - process features
    - slice proper data from data handler:  dimension of sample <feature, >
    - Build relation of samples by <time, instrument> index
        - Be able to sample times series of data <timestep, feature>
        - It will be better if the interface is like "torch.utils.data.Dataset"
    - User could build customized batch based on the data
        - The dimension of a batch of data <batch_idx, feature, timestep>
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, step_len=DEFAULT_STEP_LEN, flt_col: Optional[str] = None, **kwargs):
        self.step_len = step_len
        self.flt_col = flt_col
        super().__init__(**kwargs)

    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        self.cal = sorted(cal)

    @staticmethod
    def _extend_slice(slc: slice, cal: list, step_len: int) -> slice:
        # Dataset decide how to slice data(Get more data for timeseries).
        start, end = slc.start, slc.stop
        start_idx = bisect.bisect_left(cal, pd.Timestamp(start))
        pad_start_idx = max(0, start_idx - step_len)
        pad_start = cal[pad_start_idx]
        return slice(pad_start, end)

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        """
        split the _prepare_raw_seg is to leave a hook for data preprocessing before creating processing data
        NOTE: TSDatasetH only support slc segment on datetime !!!
        """
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        if (flt_col := kwargs.pop("flt_col", None)) is None:
            flt_col = self.flt_col

        # TSDatasetH will retrieve more data for complete time-series
        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = TSDataSampler(
            data=data,
            start=slc.start,
            end=slc.stop,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
        )
        return tsds


__all__ = ["Optional", "Dataset", "DatasetH"]
