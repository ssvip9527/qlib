import unittest
import numpy as np
from qlib.data import D
from qlib.tests import TestAutoData


class TestDataset(TestAutoData):
    def testCSI300(self):
        close_p = D.features(D.instruments("csi300"), ["$close"])
        size = close_p.groupby("datetime", group_keys=False).size()
        cnt = close_p.groupby("datetime", group_keys=False).count()["$close"]
        size_desc = size.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        cnt_desc = cnt.describe(percentiles=np.arange(0.1, 1.0, 0.1))

        print(size_desc)
        print(cnt_desc)

        self.assertLessEqual(size_desc.loc["max"], 305, "CSI300成分股数量过多")
        self.assertGreaterEqual(size_desc.loc["80%"], 290, "CSI300成分股数量不足")

        self.assertLessEqual(cnt_desc.loc["max"], 305, "CSI300成分股数量过多")
        # FIXME: 由于数据质量较低，难以确保有足够的数据
        # self.assertEqual(cnt_desc.loc["80%"], 300, "Insufficient number of CSI300 constituent stocks")

    def testClose(self):
        close_p = D.features(D.instruments("csi300"), ["Ref($close, 1)/$close - 1"])
        close_desc = close_p.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        print(close_desc)
        self.assertLessEqual(abs(close_desc.loc["90%"][0]), 0.1, "收盘价异常")
        self.assertLessEqual(abs(close_desc.loc["10%"][0]), 0.1, "收盘价异常")
        # FIXME: Yahoo数据并不完美，我们不得不
        # self.assertLessEqual(abs(close_desc.loc["max"][0]), 0.2, "Close value is abnormal")
        # self.assertGreaterEqual(close_desc.loc["min"][0], -0.2, "Close value is abnormal")


if __name__ == "__main__":
    unittest.main()
