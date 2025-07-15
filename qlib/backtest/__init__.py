# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple, Union

import pandas as pd

from .account import Account

if TYPE_CHECKING:
    from ..strategy.base import BaseStrategy
    from .executor import BaseExecutor
    from .decision import BaseTradeDecision

from ..config import C
from ..log import get_module_logger
from ..utils import init_instance_by_config
from .backtest import INDICATOR_METRIC, PORT_METRIC, backtest_loop, collect_data_loop
from .decision import Order
from .exchange import Exchange
from .utils import CommonInfrastructure

# make import more user-friendly by adding `from qlib.backtest import STH`


logger = get_module_logger("backtest caller")


def get_exchange(
    exchange: Union[str, dict, object, Path] = None,
    freq: str = "day",
    start_time: Union[pd.Timestamp, str] = None,
    end_time: Union[pd.Timestamp, str] = None,
    codes: Union[list, str] = "all",
    subscribe_fields: list = [],
    open_cost: float = 0.0015,
    close_cost: float = 0.0025,
    min_cost: float = 5.0,
    limit_threshold: Union[Tuple[str, str], float, None] | None = None,
    deal_price: Union[str, Tuple[str, str], List[str]] | None = None,
    **kwargs: Any,
) -> Exchange:
    """获取交易所实例

    参数
    ----------

    # 交易所相关参数
    exchange: Exchange
        可以是None或任何能被`init_instance_by_config`接受的类型
    freq: str
        数据频率
    start_time: Union[pd.Timestamp, str]
        回测开始时间(闭区间)
    end_time: Union[pd.Timestamp, str]
        回测结束时间(闭区间)
    codes: Union[list, str]
        股票代码列表或字符串形式的标的集合(如all, csi500, sse50)
    subscribe_fields: list
        订阅的字段列表
    open_cost : float
        开仓交易成本比例，与订单成交金额成比例
    close_cost : float
        平仓交易成本比例，与订单成交金额成比例
    min_cost : float
        最低交易成本(绝对值)，无论订单金额多少都至少收取的费用
        例如：无论订单金额多少，至少收取5元手续费
    deal_price: Union[str, Tuple[str, str], List[str]]
                支持以下两种输入格式：
                - <deal_price> : str
                - (<buy_price>, <sell_price>): Tuple[str, str] 或 List[str]

            <deal_price>, <buy_price> 或 <sell_price> := <price>
            <price> := str
            - 例如 '$close', '$open', '$vwap' (直接写"close"也可以，`Exchange`会自动添加"$"前缀)
    limit_threshold : float
        涨跌停限制比例，例如0.1表示10%，多头和空头使用相同的限制

    返回值
    -------
    :class: Exchange
    初始化后的Exchange对象
    """

    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if exchange is None:
        logger.info("Create new exchange")

        exchange = Exchange(
            freq=freq,
            start_time=start_time,
            end_time=end_time,
            codes=codes,
            deal_price=deal_price,
            subscribe_fields=subscribe_fields,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            **kwargs,
        )
        return exchange
    else:
        return init_instance_by_config(exchange, accept_types=Exchange)


def create_account_instance(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    benchmark: Optional[str],
    account: Union[float, int, dict],
    pos_type: str = "Position",
) -> Account:
    """
    # TODO: 在account中传递benchmark_config很奇怪(可能是为了报告)
    # 应该有一个后处理步骤来处理报告

    参数
    ----------
    start_time
        基准开始时间
    end_time
        基准结束时间
    benchmark : str
        用于报告的基准
    account :   Union[
                    float,
                    {
                        "cash": float,
                        "stock1": Union[
                                        int,    # 等同于{"amount": int}
                                        {"amount": int, "price"(可选): float},
                                  ]
                    },
                ]
        描述如何创建账户的信息
        对于`float`:
            仅使用初始现金创建Account
        对于`dict`:
            键"cash"表示初始现金
            键"stock1"表示第一支股票的信息，包括数量和价格(可选)
            ...
    pos_type: str
        持仓类型
    """
    if isinstance(account, (int, float)):
        init_cash = account
        position_dict = {}
    elif isinstance(account, dict):
        init_cash = account.pop("cash")
        position_dict = account
    else:
        raise ValueError("account must be in (int, float, dict)")

    return Account(
        init_cash=init_cash,
        position_dict=position_dict,
        pos_type=pos_type,
        benchmark_config=(
            {}
            if benchmark is None
            else {
                "benchmark": benchmark,
                "start_time": start_time,
                "end_time": end_time,
            }
        ),
    )


def get_strategy_executor(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    strategy: Union[str, dict, object, Path],
    executor: Union[str, dict, object, Path],
    benchmark: Optional[str] = "SH000300",
    account: Union[float, int, dict] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
) -> Tuple[BaseStrategy, BaseExecutor]:
    # 注意:
    # - 为了避免循环导入
    # - 类型注解不可靠
    from ..strategy.base import BaseStrategy  # pylint: disable=C0415
    from .executor import BaseExecutor  # pylint: disable=C0415

    trade_account = create_account_instance(
        start_time=start_time,
        end_time=end_time,
        benchmark=benchmark,
        account=account,
        pos_type=pos_type,
    )

    exchange_kwargs = copy.copy(exchange_kwargs)
    if "start_time" not in exchange_kwargs:
        exchange_kwargs["start_time"] = start_time
    if "end_time" not in exchange_kwargs:
        exchange_kwargs["end_time"] = end_time
    trade_exchange = get_exchange(**exchange_kwargs)

    common_infra = CommonInfrastructure(trade_account=trade_account, trade_exchange=trade_exchange)
    trade_strategy = init_instance_by_config(strategy, accept_types=BaseStrategy)
    trade_strategy.reset_common_infra(common_infra)
    trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor)
    trade_executor.reset_common_infra(common_infra)

    return trade_strategy, trade_executor


def backtest(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    strategy: Union[str, dict, object, Path],
    executor: Union[str, dict, object, Path],
    benchmark: str = "SH000300",
    account: Union[float, int, dict] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
) -> Tuple[PORT_METRIC, INDICATOR_METRIC]:
    """initialize the strategy and executor, then backtest function for the interaction of the outermost strategy and
    executor in the nested decision execution

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[pd.Timestamp, str]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]),   setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    strategy : Union[str, dict, object, Path]
        for initializing outermost portfolio strategy. Please refer to the docs of init_instance_by_config for more
        information.
    executor : Union[str, dict, object, Path]
        for initializing the outermost executor.
    benchmark: str
        the benchmark for reporting.
    account : Union[float, int, Position]
        information for describing how to create the account
        For `float` or `int`:
            Using Account with only initial cash
        For `Position`:
            Using Account with a Position
    exchange_kwargs : dict
        the kwargs for initializing Exchange
    pos_type : str
        the type of Position.

    Returns
    -------
    portfolio_dict: PORT_METRIC
        it records the trading portfolio_metrics information
    indicator_dict: INDICATOR_METRIC
        it computes the trading indicator
        It is organized in a dict format

    """
    trade_strategy, trade_executor = get_strategy_executor(
        start_time,
        end_time,
        strategy,
        executor,
        benchmark,
        account,
        exchange_kwargs,
        pos_type=pos_type,
    )
    return backtest_loop(start_time, end_time, trade_strategy, trade_executor)


def collect_data(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    strategy: Union[str, dict, object, Path],
    executor: Union[str, dict, object, Path],
    benchmark: str = "SH000300",
    account: Union[float, int, dict] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
    return_value: dict | None = None,
) -> Generator[object, None, None]:
    """initialize the strategy and executor, then collect the trade decision data for rl training

    please refer to the docs of the backtest for the explanation of the parameters

    Yields
    -------
    object
        trade decision
    """
    trade_strategy, trade_executor = get_strategy_executor(
        start_time,
        end_time,
        strategy,
        executor,
        benchmark,
        account,
        exchange_kwargs,
        pos_type=pos_type,
    )
    yield from collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value=return_value)


def format_decisions(
    decisions: List[BaseTradeDecision],
) -> Optional[Tuple[str, List[Tuple[BaseTradeDecision, Union[Tuple, None]]]]]:
    """
    格式化由`qlib.backtest.collect_data`收集的决策数据
    决策将被组织成树状结构

    参数
    ----------
    decisions : List[BaseTradeDecision]
        由`qlib.backtest.collect_data`收集的决策列表

    返回值
    -------
    Tuple[str, List[Tuple[BaseTradeDecision, Union[Tuple, None]]]:

        将决策列表重新格式化为更易读的形式
        <decisions> :=  Tuple[<频率>, List[Tuple[<决策>, <子决策>]]
        - <sub decisions> := `<低层级的decisions>` | None
        - <freq> := "day" | "30min" | "1min" | ...
        - <decision> := <BaseTradeDecision的实例>
    """
    if len(decisions) == 0:
        return None

    cur_freq = decisions[0].strategy.trade_calendar.get_freq()

    res: Tuple[str, list] = (cur_freq, [])
    last_dec_idx = 0
    for i, dec in enumerate(decisions[1:], 1):
        if dec.strategy.trade_calendar.get_freq() == cur_freq:
            res[1].append((decisions[last_dec_idx], format_decisions(decisions[last_dec_idx + 1 : i])))
            last_dec_idx = i
    res[1].append((decisions[last_dec_idx], format_decisions(decisions[last_dec_idx + 1 :])))
    return res


__all__ = ["Order", "backtest", "get_strategy_executor"]
