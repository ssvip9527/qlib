# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Generator, Optional, Tuple, Union, cast

import pandas as pd

from qlib.backtest.decision import BaseTradeDecision
from qlib.backtest.report import Indicator

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.executor import BaseExecutor

from tqdm.auto import tqdm

from ..utils.time import Freq


PORT_METRIC = Dict[str, Tuple[pd.DataFrame, dict]]
INDICATOR_METRIC = Dict[str, Tuple[pd.DataFrame, Indicator]]


def backtest_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
) -> Tuple[PORT_METRIC, INDICATOR_METRIC]:
    """嵌套决策执行中最外层策略与执行器交互的回测函数

    请参考 `collect_data_loop` 的文档

    返回值
    -------
    portfolio_dict: PORT_METRIC
        记录交易组合指标信息
    indicator_dict: INDICATOR_METRIC
        计算交易指标
    """
    return_value: dict = {}
    for _decision in collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value):
        pass

    portfolio_dict = cast(PORT_METRIC, return_value.get("portfolio_dict"))
    indicator_dict = cast(INDICATOR_METRIC, return_value.get("indicator_dict"))

    return portfolio_dict, indicator_dict


def collect_data_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
    return_value: dict | None = None,
) -> Generator[BaseTradeDecision, Optional[BaseTradeDecision], None]:
    """用于收集强化学习训练所需交易决策数据的生成器

    参数
    ----------
    start_time : Union[pd.Timestamp, str]
        回测的闭区间开始时间
        **注意**: 此时间将应用于最外层执行器的日历。
    end_time : Union[pd.Timestamp, str]
        回测的闭区间结束时间
        **注意**: 此时间将应用于最外层执行器的日历。
        例如：Executor[day](Executor[1min])，设置`end_time == 20XX0301`将包含20XX0301当天的所有分钟数据
    trade_strategy : BaseStrategy
        最外层的组合策略
    trade_executor : BaseExecutor
        最外层的执行器
    return_value : dict
        用于backtest_loop函数

    生成值
    -------
    object
        交易决策
    """
    trade_executor.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(level_infra=trade_executor.get_level_infra())

    with tqdm(total=trade_executor.trade_calendar.get_trade_len(), desc="backtest loop") as bar:
        _execute_result = None
        while not trade_executor.finished():
            _trade_decision: BaseTradeDecision = trade_strategy.generate_trade_decision(_execute_result)
            _execute_result = yield from trade_executor.collect_data(_trade_decision, level=0)
            trade_strategy.post_exe_step(_execute_result)
            bar.update(1)
        trade_strategy.post_upper_level_exe_step()

    if return_value is not None:
        all_executors = trade_executor.get_all_executors()

        portfolio_dict: PORT_METRIC = {}
        indicator_dict: INDICATOR_METRIC = {}

        for executor in all_executors:
            key = "{}{}".format(*Freq.parse(executor.time_per_step))
            if executor.trade_account.is_port_metr_enabled():
                portfolio_dict[key] = executor.trade_account.get_portfolio_metrics()

            indicator_df = executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
            indicator_obj = executor.trade_account.get_trade_indicator()
            indicator_dict[key] = (indicator_df, indicator_obj)

        return_value.update({"portfolio_dict": portfolio_dict, "indicator_dict": indicator_dict})
