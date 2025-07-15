# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, cast

import pandas as pd

from qlib.utils import init_instance_by_config

from .decision import BaseTradeDecision, Order
from .exchange import Exchange
from .high_performance_ds import BaseOrderIndicator
from .position import BasePosition
from .report import Indicator, PortfolioMetrics

"""
账户中的rtn（收益）与earning（盈利）说明
    rtn:
        从订单视角出发
        1. 当有任何订单执行时变化（卖单或买单）
        2. 在当日结束时变化，计算公式为 (当日收盘价 - 股票价格) * 数量
    earning
        从当前持仓价值出发
        在交易日结束时更新
        盈利 = 当日持仓价值 - 前日持仓价值
    **是否考虑成本**
        earning是两个持仓价值的差值，因此已考虑成本，代表真实回报率
        而rtn的具体实现中未考虑成本，即 rtn - 成本 = earning

"""


class AccumulatedInfo:
    """
    累积交易信息，包括累积收益/成本/成交额
    AccumulatedInfo 应在不同层级间共享
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rtn: float = 0.0  # 累积收益，不考虑成本
        self.cost: float = 0.0  # 累积成本
        self.to: float = 0.0  # 累积成交额

    def add_return_value(self, value: float) -> None:
        self.rtn += value

    def add_cost(self, value: float) -> None:
        self.cost += value

    def add_turnover(self, value: float) -> None:
        self.to += value

    @property
    def get_return(self) -> float:
        return self.rtn

    @property
    def get_cost(self) -> float:
        return self.cost

    @property
    def get_turnover(self) -> float:
        return self.to


class Account:
    """
    嵌套执行中Account指标的正确性依赖于qlib/backtest/executor.py:NestedExecutor中`trade_account`的浅拷贝
    不同层级的执行器在计算指标时拥有不同的Account对象，但持仓对象在所有Account对象间共享。
    """

    def __init__(
        self,
        init_cash: float = 1e9,
        position_dict: dict = {},
        freq: str = "day",
        benchmark_config: dict = {},
        pos_type: str = "Position",
        port_metr_enabled: bool = True,
    ) -> None:
        """回测的交易账户。

        参数
        ----------
        init_cash : float, optional
            初始现金，默认为1e9
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # 等同于 {"amount": int}
                                {"amount": int, "price"(可选): float},
                            ]
                        ]
            包含数量和价格参数的初始股票，
            如果股票字典中没有price键，将通过_fill_stock_value方法填充。
            默认为空字典。
        """

        self._pos_type = pos_type
        self._port_metr_enabled = port_metr_enabled
        self.benchmark_config: dict = {}  # avoid no attribute error
        self.init_vars(init_cash, position_dict, freq, benchmark_config)

    def init_vars(self, init_cash: float, position_dict: dict, freq: str, benchmark_config: dict) -> None:
        # 1) 以下变量由多个层级共享
        # - 在NestedExecutor中会看到浅拷贝而非深拷贝
        self.init_cash = init_cash
        self.current_position: BasePosition = init_instance_by_config(
            {
                "class": self._pos_type,
                "kwargs": {
                    "cash": init_cash,
                    "position_dict": position_dict,
                },
                "module_path": "qlib.backtest.position",
            },
        )
        self.accum_info = AccumulatedInfo()

        # 2) following variables are not shared between layers
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        self.hist_positions: Dict[pd.Timestamp, BasePosition] = {}
        self.reset(freq=freq, benchmark_config=benchmark_config)

    def is_port_metr_enabled(self) -> bool:
        """
        是否启用基于投资组合的指标。
        """
        return self._port_metr_enabled and not self.current_position.skip_update()

    def reset_report(self, freq: str, benchmark_config: dict) -> None:
        # portfolio related metrics
        if self.is_port_metr_enabled():
            # 注意:
            # `accum_info`和`current_position`在这里是共享的
            self.portfolio_metrics = PortfolioMetrics(freq, benchmark_config)
            self.hist_positions = {}

            # 填充股票价值
            # 账户频率可能与交易频率不一致
            # 当数据质量较低时，这可能导致难以发现的错误
            if isinstance(self.benchmark_config, dict) and "start_time" in self.benchmark_config:
                self.current_position.fill_stock_value(self.benchmark_config["start_time"], self.freq)

        # trading related metrics(e.g. high-frequency trading)
        self.indicator = Indicator()

    def reset(
        self, freq: str | None = None, benchmark_config: dict | None = None, port_metr_enabled: bool | None = None
    ) -> None:
        """
        重置账户的频率和报告

        参数
        ----------
        freq : str
            账户和报告的频率，默认为None
        benchmark_config : {}, optional
            报告的基准配置，默认为None
        port_metr_enabled: bool
            是否启用投资组合指标
        """
        if freq is not None:
            self.freq = freq
        if benchmark_config is not None:
            self.benchmark_config = benchmark_config
        if port_metr_enabled is not None:
            self._port_metr_enabled = port_metr_enabled

        self.reset_report(self.freq, self.benchmark_config)

    def get_hist_positions(self) -> Dict[pd.Timestamp, BasePosition]:
        return self.hist_positions

    def get_cash(self) -> float:
        return self.current_position.get_cash()

    def _update_state_from_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        if self.is_port_metr_enabled():
            # 更新换手率
            self.accum_info.add_turnover(trade_val)
            # 更新成本
            self.accum_info.add_cost(cost)

            # 根据订单更新收益
            trade_amount = trade_val / trade_price
            if order.direction == Order.SELL:  # 0 for sell
                # 卖出股票时，从价格变动中获取利润
                profit = trade_val - self.current_position.get_stock_price(order.stock_id) * trade_amount
                self.accum_info.add_return_value(profit)  # 注意此处不考虑成本

            elif order.direction == Order.BUY:  # 1 for buy
                # 买入股票时，为收益率计算方法获取收益
                # 买入订单的利润是为了使收益率与bar结束时的收益保持一致
                profit = self.current_position.get_stock_price(order.stock_id) * trade_amount - trade_val
                self.accum_info.add_return_value(profit)  # 注意此处不考虑成本

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        if self.current_position.skip_update():
            # TODO: 支持账户的多态性
            # 对无限持仓更新订单没有意义
            return

        # 如果股票已卖出，Position中没有股票价格信息，则应先更新账户
        # 然后更新当前持仓
        # 如果股票被买入，当前持仓中没有该股票，先更新持仓，再更新账户
        # 成本将在最后从现金中扣除，因此交易逻辑可以忽略成本计算
        if order.direction == Order.SELL:
            # sell stock
            self._update_state_from_order(order, trade_val, cost, trade_price)
            # update current position
            # for may sell all of stock_id
            self.current_position.update_order(order, trade_val, cost, trade_price)
        else:
            # buy stock
            # deal order, then update state
            self.current_position.update_order(order, trade_val, cost, trade_price)
            self._update_state_from_order(order, trade_val, cost, trade_price)

    def update_current_position(
        self,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        trade_exchange: Exchange,
    ) -> None:
        """
        更新当前状态以使收益率与bar结束时的收益一致，并更新股票的持有bar计数
        """
        # 更新持仓中股票的价格和价格变动带来的利润
        # 注意：更新持仓不仅服务于投资组合指标，也服务于策略
        assert self.current_position is not None

        if not self.current_position.skip_update():
            stock_list = self.current_position.get_stock_list()
            for code in stock_list:
                # if suspended, no new price to be updated, profit is 0
                if trade_exchange.check_stock_suspended(code, trade_start_time, trade_end_time):
                    continue
                bar_close = cast(float, trade_exchange.get_close(code, trade_start_time, trade_end_time))
                self.current_position.update_stock_price(stock_id=code, price=bar_close)
            # 更新持仓天数计数
            # 注意：更新bar计数不仅服务于投资组合指标，也服务于策略
            self.current_position.add_count_all(bar=self.freq)

    def update_portfolio_metrics(self, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> None:
        """更新投资组合指标"""
        # 计算收益
        # 账户价值 - 上次账户价值
        # 对于第一个交易日，账户价值 - 初始现金
        # 使用self.portfolio_metrics.is_empty()判断是否为第一个交易日
        # 获取上次账户价值、上次总成本和上次总换手率
        assert self.portfolio_metrics is not None

        if self.portfolio_metrics.is_empty():
            last_account_value = self.init_cash
            last_total_cost = 0
            last_total_turnover = 0
        else:
            last_account_value = self.portfolio_metrics.get_latest_account_value()
            last_total_cost = self.portfolio_metrics.get_latest_total_cost()
            last_total_turnover = self.portfolio_metrics.get_latest_total_turnover()

        # get now_account_value, now_stock_value, now_earning, now_cost, now_turnover
        now_account_value = self.current_position.calculate_value()
        now_stock_value = self.current_position.calculate_stock_value()
        now_earning = now_account_value - last_account_value
        now_cost = self.accum_info.get_cost - last_total_cost
        now_turnover = self.accum_info.get_turnover - last_total_turnover

        # 更新今天的投资组合指标
        # 判断交易是否开始
        # 不要将初始账户状态添加到portfolio_metrics中，因为那些天我们没有超额收益
        self.portfolio_metrics.update_portfolio_metrics_record(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            account_value=now_account_value,
            cash=self.current_position.position["cash"],
            return_rate=(now_earning + now_cost) / last_account_value,
            # 这里使用收益来计算回报率，持仓视角，收益考虑了成本，是真实回报
            # 为了与evaluate.py中的原始回测定义保持一致
            total_turnover=self.accum_info.get_turnover,
            turnover_rate=now_turnover / last_account_value,
            total_cost=self.accum_info.get_cost,
            cost_rate=now_cost / last_account_value,
            stock_value=now_stock_value,
        )

    def update_hist_positions(self, trade_start_time: pd.Timestamp) -> None:
        """更新历史持仓"""
        now_account_value = self.current_position.calculate_value()
        # set now_account_value to position
        self.current_position.position["now_account_value"] = now_account_value
        self.current_position.update_weight_all()
        # 更新历史持仓
        # 注意使用深拷贝
        self.hist_positions[trade_start_time] = copy.deepcopy(self.current_position)

    def update_indicator(
        self,
        trade_start_time: pd.Timestamp,
        trade_exchange: Exchange,
        atomic: bool,
        outer_trade_decision: BaseTradeDecision,
        trade_info: list = [],
        inner_order_indicators: List[BaseOrderIndicator] = [],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = [],
        indicator_config: dict = {},
    ) -> None:
        """在每个bar结束时更新交易指标和订单指标"""
        # TODO: 跳过空决策会更快吗？`outer_trade_decision.empty()`

        # 指标是与交易（例如高频订单执行）相关的分析
        self.indicator.reset()

        # 聚合每个订单的信息
        if atomic:
            self.indicator.update_order_indicators(trade_info)
        else:
            self.indicator.agg_order_indicators(
                inner_order_indicators,
                decision_list=decision_list,
                outer_trade_decision=outer_trade_decision,
                trade_exchange=trade_exchange,
                indicator_config=indicator_config,
            )

        # 一次性聚合所有订单指标
        self.indicator.cal_trade_indicators(trade_start_time, self.freq, indicator_config)

        # 记录指标
        self.indicator.record(trade_start_time)

    def update_bar_end(
        self,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        trade_exchange: Exchange,
        atomic: bool,
        outer_trade_decision: BaseTradeDecision,
        trade_info: list = [],
        inner_order_indicators: List[BaseOrderIndicator] = [],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = [],
        indicator_config: dict = {},
    ) -> None:
        """
        在每个交易周期结束时更新账户

        参数
        ----------
        trade_start_time : pd.Timestamp
            周期的闭区间开始时间
        trade_end_time : pd.Timestamp
            周期的闭区间结束时间
        trade_exchange : Exchange
            交易交易所，用于更新当前状态
        atomic : bool
            交易执行器是否为原子执行器，即内部没有更高频率的交易执行器
            - 如果atomic为True，使用trade_info计算指标
            - 否则，聚合内部指标
        outer_trade_decision: BaseTradeDecision
            外部交易决策
        trade_info : List[(Order, float, float, float)], optional
            交易信息，默认为None
            - 当atomic为True时必需
            - 元组列表(order, 交易价值, 交易成本, 交易价格)
        inner_order_indicators : Indicator, optional
            内部执行器的指标，默认为None
            - 当atomic为False时必需
            - 用于聚合外部指标
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = None,
            内部层级的决策列表：List[Tuple[<决策>, <开始时间>, <结束时间>]]
            内部层级
        indicator_config : dict, optional
            计算指标的配置，默认为{}
        """
        if atomic is True and trade_info is None:
            raise ValueError("trade_info is necessary in atomic executor")
        elif atomic is False and inner_order_indicators is None:
            raise ValueError("inner_order_indicators is necessary in un-atomic executor")

        # 在每个bar结束时更新当前持仓和持有bar计数
        self.update_current_position(trade_start_time, trade_end_time, trade_exchange)

        if self.is_port_metr_enabled():
            # portfolio_metrics是投资组合相关分析
            self.update_portfolio_metrics(trade_start_time, trade_end_time)
            self.update_hist_positions(trade_start_time)

        # 在每个bar结束时更新指标
        self.update_indicator(
            trade_start_time=trade_start_time,
            trade_exchange=trade_exchange,
            atomic=atomic,
            outer_trade_decision=outer_trade_decision,
            trade_info=trade_info,
            inner_order_indicators=inner_order_indicators,
            decision_list=decision_list,
            indicator_config=indicator_config,
        )

    def get_portfolio_metrics(self) -> Tuple[pd.DataFrame, dict]:
        """获取历史投资组合指标和持仓实例"""
        if self.is_port_metr_enabled():
            assert self.portfolio_metrics is not None
            _portfolio_metrics = self.portfolio_metrics.generate_portfolio_metrics_dataframe()
            _positions = self.get_hist_positions()
            return _portfolio_metrics, _positions
        else:
            raise ValueError("generate_portfolio_metrics should be True if you want to generate portfolio_metrics")

    def get_trade_indicator(self) -> Indicator:
        """获取交易指标实例，包含pa/pos/ffr信息。"""
        return self.indicator
