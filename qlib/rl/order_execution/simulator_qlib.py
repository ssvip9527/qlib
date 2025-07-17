# 版权所有 (c) 微软公司。
# MIT许可证授权。

from __future__ import annotations

from typing import Generator, List, Optional

import pandas as pd

from qlib.backtest import collect_data_loop, get_strategy_executor
from qlib.backtest.decision import BaseTradeDecision, Order, TradeRangeByTime
from qlib.backtest.executor import NestedExecutor
from qlib.rl.data.integration import init_qlib
from qlib.rl.simulator import Simulator
from .state import SAOEState
from .strategy import SAOEStateAdapter, SAOEStrategy


class SingleAssetOrderExecution(Simulator[Order, SAOEState, float]):
    """基于Qlib回测工具实现的单一资产订单执行(SAOE)模拟器。

    参数
    ----------
    order
        启动SAOE模拟器的种子是一个订单。
    executor_config
        执行器配置
    exchange_config
        交易所配置
    qlib_config
        用于初始化Qlib的配置。如果为None，则不会初始化Qlib。
    cash_limit:
        现金限制。
    """

    def __init__(
        self,
        order: Order,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict | None = None,
        cash_limit: float | None = None,
    ) -> None:
        super().__init__(initial=order)

        assert order.start_time.date() == order.end_time.date(), "Start date and end date must be the same."

        strategy_config = {
            "class": "SingleOrderStrategy",
            "module_path": "qlib.rl.strategy.single_order",
            "kwargs": {
                "order": order,
                "trade_range": TradeRangeByTime(order.start_time.time(), order.end_time.time()),
            },
        }

        self._collect_data_loop: Optional[Generator] = None
        self.reset(order, strategy_config, executor_config, exchange_config, qlib_config, cash_limit)

    def reset(
        self,
        order: Order,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict | None = None,
        cash_limit: Optional[float] = None,
    ) -> None:
        if qlib_config is not None:
            init_qlib(qlib_config)

        strategy, self._executor = get_strategy_executor(
            start_time=order.date,
            end_time=order.date + pd.DateOffset(1),
            strategy=strategy_config,
            executor=executor_config,
            benchmark=order.stock_id,
            account=cash_limit if cash_limit is not None else int(1e12),
            exchange_kwargs=exchange_config,
            pos_type="Position" if cash_limit is not None else "InfPosition",
        )

        assert isinstance(self._executor, NestedExecutor)

        self.report_dict: dict = {}
        self.decisions: List[BaseTradeDecision] = []
        self._collect_data_loop = collect_data_loop(
            start_time=order.date,
            end_time=order.date,
            trade_strategy=strategy,
            trade_executor=self._executor,
            return_value=self.report_dict,
        )
        assert isinstance(self._collect_data_loop, Generator)

        self.step(action=None)

        self._order = order

    def _get_adapter(self) -> SAOEStateAdapter:
        return self._last_yielded_saoe_strategy.adapter_dict[self._order.key_by_day]

    @property
    def twap_price(self) -> float:
        return self._get_adapter().twap_price

    def _iter_strategy(self, action: Optional[float] = None) -> SAOEStrategy:
        """Iterate the _collect_data_loop until we get the next yield SAOEStrategy."""
        assert self._collect_data_loop is not None

        obj = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        while not isinstance(obj, SAOEStrategy):
            if isinstance(obj, BaseTradeDecision):
                self.decisions.append(obj)
            obj = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        assert isinstance(obj, SAOEStrategy)
        return obj

    def step(self, action: Optional[float]) -> None:
        """执行一步SAOE操作。

        参数
        ----------
        action (float):
            希望成交的数量。模拟器不保证所有数量都能成功成交。
        """

        assert not self.done(), "Simulator has already done!"

        try:
            self._last_yielded_saoe_strategy = self._iter_strategy(action=action)
        except StopIteration:
            pass

        assert self._executor is not None

    def get_state(self) -> SAOEState:
        return self._get_adapter().saoe_state

    def done(self) -> bool:
        return self._executor.finished()
