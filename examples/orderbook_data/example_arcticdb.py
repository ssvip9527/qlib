# 版权所有 (c) 微软公司。
# 根据 MIT 许可证授权。

# 使用ArcticDB替代Arctic
import arcticdb as adb
import qlib
from qlib.data import D
import unittest


class TestClass(unittest.TestCase):
    """
    有用的命令
    - 运行所有测试: pytest examples/orderbook_data/example_arcticdb.py
    - 运行单个测试: pytest -s --pdb --disable-warnings examples/orderbook_data/example_arcticdb.py::TestClass::test_basic01
    """

    def setUp(self):
        """
        为ArcticDB进行配置
        """
        provider_uri = "~/.qlib/qlib_data/yahoo_cn_1min"
        qlib.init(
            provider_uri=provider_uri,
            mem_cache_size_limit=1024**3 * 2,
            mem_cache_type="sizeof",
            kernels=1,
            expression_provider={"class": "LocalExpressionProvider", "kwargs": {"time2idx": False}},
            feature_provider={
                "class": "ArcticDBFeatureProvider",
                "module_path": "qlib.contrib.data.arcticdb_data",
                "kwargs": {"uri": "lmdb:///tmp/arcticdb"},
            },
            dataset_provider={
                "class": "LocalDatasetProvider",
                "kwargs": {
                    "align_time": False,  # Order book is not fixed, so it can't be align to a shared fixed frequency calendar
                },
            },
        )
        # self.stocks_list = ["SH600519"]
        self.stocks_list = ["SZ000725"]

    def test_basic(self):
        # NOTE: this data contains a lot of zeros in $askX and $bidX
        df = D.features(
            self.stocks_list,
            fields=["$ask1", "$ask2", "$bid1", "$bid2"],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic_without_time(self):
        df = D.features(self.stocks_list, fields=["$ask1"], freq="ticks")
        print(df)

    def test_basic01(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic02(self):
        df = D.features(
            self.stocks_list,
            fields=["$function_code"],
            freq="transaction",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic03(self):
        df = D.features(
            self.stocks_list,
            fields=["$function_code"],
            freq="order",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    # 这里是一些常用的高频表达式
    # 1) 一些共享表达式
    expr_sum_buy_ask_1 = "(TResample($ask1, '1min', 'last') + TResample($bid1, '1min', 'last'))"
    total_volume = (
        "TResample("
        + "+".join([f"${name}{i}" for i in range(1, 11) for name in ["asize", "bsize"]])
        + ", '1min', 'sum')"
    )

    @staticmethod
    def total_func(name, method):
        return "TResample(" + "+".join([f"${name}{i}" for i in range(1, 11)]) + ",'1min', '{}'".format(method)

    def test_basic04(self):
        df = D.features(
            self.stocks_list,
            fields=[self.expr_sum_buy_ask_1],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic05(self):
        df = D.features(
            self.stocks_list,
            fields=[self.total_volume],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic06(self):
        df = D.features(
            self.stocks_list,
            fields=[self.total_func("asize", "sum")],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic07(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic08(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic09(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic10(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic11(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic12(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic13(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic14(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic15(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic16(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic17(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic18(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic19(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)

    def test_basic20(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230 09:30:00",
            end_time="20201230 10:30:00",
        )
        print(df)