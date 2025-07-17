# 版权所有 (c) 微软公司。
# MIT许可证授权。

from __future__ import annotations

from qlib.backtest import Order
from qlib.backtest.decision import OrderHelper, TradeDecisionWO, TradeRange
from qlib.strategy.base import BaseStrategy


class SingleOrderStrategy(BaseStrategy):
    """用于生成只包含一个订单的交易决策的策略。"""

    def __init__(
        self,
        order: Order,
        trade_range: TradeRange | None = None,
    ) -> None:
        super().__init__()

        self._order = order
        self._trade_range = trade_range

    def generate_trade_decision(self, execute_result: list | None = None) -> TradeDecisionWO:
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        order_list = [
            oh.create(
                code=self._order.stock_id,
                amount=self._order.amount,
                direction=self._order.direction,
            ),
        ]
        return TradeDecisionWO(order_list, self, self._trade_range)
