# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import pytest
import sys
from qlib.tests import TestAutoData
from qlib.data.dataset import TSDatasetH, TSDataSampler
import numpy as np
import pandas as pd
import time
from qlib.data.dataset.handler import DataHandlerLP


class TestDataset(TestAutoData):
    @pytest.mark.slow
    def testTSDataset(self):
        tsdh = TSDatasetH(
            handler={
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2017-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2017-01-01",
                    "fit_end_time": "2017-12-31",
                    "instruments": "csi300",
                    "infer_processors": [
                        {"class": "FilterCol", "kwargs": {"col_list": ["RESI5", "WVMA5", "RSQR5"]}},
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": "true"}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        "DropnaLabel",
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},  # 截面排序归一化
                    ],
                },
            },
            segments={
                "train": ("2017-01-01", "2017-12-31"),
                "valid": ("2018-01-01", "2018-12-31"),
                "test": ("2019-01-01", "2020-08-01"),
            },
        )
        tsds_train = tsdh.prepare("train", data_key=DataHandlerLP.DK_L)  # 测试正确性
        tsds = tsdh.prepare("valid", data_key=DataHandlerLP.DK_L)

        t = time.time()
        for idx in np.random.randint(0, len(tsds_train), size=2000):
            _ = tsds_train[idx]
        print(f"2000 sample takes {time.time() - t}s")

        t = time.time()
        for _ in range(20):
            data = tsds_train[np.random.randint(0, len(tsds_train), size=2000)]
        print(data.shape)
        print(f"2000 sample(batch index) * 20 times takes {time.time() - t}s")

        # 样本维度与表格数据相同，但返回的是样本的时间序列数据

        # 我们有两种方法获取样本的时间序列

        # 1) 通过整数索引直接采样
        tsds[len(tsds) - 1]

        # 2) 通过<日期时间,标的>索引采样
        data_from_ds = tsds["2017-12-31", "SZ300315"]

        # 检查数据
        # 直接从DataFrame获取数据
        data_from_df = (
            tsdh.handler.fetch(data_key=DataHandlerLP.DK_L)
            .loc(axis=0)["2017-01-01":"2017-12-31", "SZ300315"]
            .iloc[-30:]
            .values
        )

        equal = np.isclose(data_from_df, data_from_ds)
        self.assertTrue(equal[~np.isnan(data_from_df)].all())

        if False:
            # 3) 同时获取索引和数据
            # 注意：我们不想依赖pytorch，所以这个测试不能包含进来。这只是一个示例
            from torch.utils.data import DataLoader
            from qlib.model.utils import IndexSampler

            i = len(tsds) - 1
            idx = tsds.get_index()
            tsds[i]
            idx[i]

            s_w_i = IndexSampler(tsds)
            test_loader = DataLoader(s_w_i)

            s_w_i[3]
            for data, i in test_loader:
                break
            print(data.shape)
            print(idx[i])


class TestTSDataSampler(unittest.TestCase):
    def test_TSDataSampler(self):
        """
        测试TSDataSampler以解决#1716问题
        """
        datetime_list = ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"]
        instruments = ["000001", "000002", "000003", "000004", "000005"]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(datetime_list), instruments], names=["datetime", "instrument"]
        )
        data = np.random.randn(len(datetime_list) * len(instruments))
        test_df = pd.DataFrame(data=data, index=index, columns=["factor"])
        dataset = TSDataSampler(test_df, datetime_list[0], datetime_list[-1], step_len=2)
        print()
        print("--------------dataset[0]--------------")
        print(dataset[0])
        print("--------------dataset[1]--------------")
        print(dataset[1])
        assert len(dataset[0]) == 2
        self.assertTrue(np.isnan(dataset[0][0]))
        self.assertEqual(dataset[0][1], dataset[1][0])
        self.assertEqual(dataset[1][1], dataset[2][0])
        self.assertEqual(dataset[2][1], dataset[3][0])

    def test_TSDataSampler2(self):
        """
        额外测试TSDataSampler以防止对前面的值进行错误的nan填充
        """
        datetime_list = ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"]
        instruments = ["000001", "000002", "000003", "000004", "000005"]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(datetime_list), instruments], names=["datetime", "instrument"]
        )
        data = np.random.randn(len(datetime_list) * len(instruments))
        test_df = pd.DataFrame(data=data, index=index, columns=["factor"])
        dataset = TSDataSampler(test_df, datetime_list[2], datetime_list[-1], step_len=3)
        print()
        print("--------------dataset[0]--------------")
        print(dataset[0])
        print("--------------dataset[1]--------------")
        print(dataset[1])
        for i in range(3):
            self.assertFalse(np.isnan(dataset[0][i]))
            self.assertFalse(np.isnan(dataset[1][i]))
        self.assertEqual(dataset[0][1], dataset[1][0])
        self.assertEqual(dataset[0][2], dataset[1][1])


if __name__ == "__main__":
    unittest.main(verbosity=10)

    # User could use following code to run test when using line_profiler
    # td = TestDataset()
    # td.setUpClass()
    # td.testTSDataset()
