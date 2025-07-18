# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from qlib.backtest import backtest
from qlib.tests import TestAutoData
import pandas as pd
from pathlib import Path
from qlib.data import D
import numpy as np

DIRNAME = Path(__file__).absolute().resolve().parent


class FileStrTest(TestAutoData):
    # 保证测试正确性的假设
    # - 这些交易日内没有价格复权。
    TEST_INST = "SH600519"

    EXAMPLE_FILE = DIRNAME / "order_example.csv"

    def _gen_orders(self, dealt_num_for_1000) -> pd.DataFrame:
        headers = [
            "datetime",
            "instrument",
            "amount",
            "direction",
        ]
        orders = [
            # 测试买入时现金限制
            ["20200103", self.TEST_INST, "1000", "buy"],
            # 测试买入时最小费用
            ["20200106", self.TEST_INST, "1", "buy"],
            # 测试卖出时持仓股票限制
            ["20200107", self.TEST_INST, "1000", "sell"],
            # 测试买入时现金限制
            ["20200108", self.TEST_INST, "1000", "buy"],
            # 测试卖出时最小费用
            ["20200109", self.TEST_INST, "1", "sell"],
            # 测试全部卖出股票
            ["20200110", self.TEST_INST, str(dealt_num_for_1000), "sell"],
        ]
        return pd.DataFrame(orders, columns=headers).set_index(["datetime", "instrument"])

    def test_file_str(self):
        # 0) 基本设置
        account_money = 150000

        # 1) 获取信息
        df = D.features([self.TEST_INST], ["$close", "$factor"], start_time="20200103", end_time="20200103")
        price = df["$close"].item()
        factor = df["$factor"].item()
        price_unit = price / factor * 100
        dealt_num_for_1000 = (account_money // price_unit) * (100 / factor)
        print(price, factor, price_unit, dealt_num_for_1000)

        # 2) 生成订单
        orders = self._gen_orders(dealt_num_for_1000)
        orders.to_csv(self.EXAMPLE_FILE)
        print(orders)

        # 3) 执行策略
        strategy_config = {
            "class": "FileOrderStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
            "kwargs": {"file": self.EXAMPLE_FILE},
        }

        freq = "day"
        start_time = "2020-01-01"
        end_time = "2020-01-16"
        codes = [self.TEST_INST]

        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": account_money,
            "benchmark": None,  # 这里交易不需要基准
            "exchange_kwargs": {
                "freq": freq,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 500,
                "codes": codes,
                "trade_unit": 100,
            },
            # "pos_type": "InfPosition"  # 无限持仓类型
        }
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq,
                "generate_portfolio_metrics": False,
                "verbose": True,
                "indicator_config": {
                    "show_indicator": False,
                },
            },
        }
        report_dict, indicator_dict = backtest(
            executor=executor_config,
            strategy=strategy_config,
            **backtest_config,
        )

        # ffr 校验
        ffr_dict = indicator_dict["1day"][0]["ffr"].to_dict()
        ffr_dict = {str(date).split()[0]: ffr_dict[date] for date in ffr_dict}
        assert np.isclose(ffr_dict["2020-01-03"], dealt_num_for_1000 / 1000)
        assert np.isclose(ffr_dict["2020-01-06"], 0)
        assert np.isclose(ffr_dict["2020-01-07"], dealt_num_for_1000 / 1000)
        assert np.isclose(ffr_dict["2020-01-08"], dealt_num_for_1000 / 1000)
        assert np.isclose(ffr_dict["2020-01-09"], 0)
        assert np.isclose(ffr_dict["2020-01-10"], 1)

        self.EXAMPLE_FILE.unlink()


if __name__ == "__main__":
    unittest.main()
