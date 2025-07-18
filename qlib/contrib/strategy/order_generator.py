# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权

"""
此订单生成器用于基于WeightStrategyBase的策略
"""
from ...backtest.position import Position
from ...backtest.exchange import Exchange

import pandas as pd
import copy


class OrderGenerator:
    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        """generate_order_list_from_target_weight_position

        :param current: 当前持仓
        :type current: Position
        :param trade_exchange: 交易所
        :type trade_exchange: Exchange
        :param target_weight_position: {股票ID : 权重}
        :type target_weight_position: dict
        :param risk_degree: 风险度
        :type risk_degree: float
        :param pred_start_time: 预测开始时间
        :type pred_start_time: pd.Timestamp
        :param pred_end_time: 预测结束时间
        :type pred_end_time: pd.Timestamp
        :param trade_start_time: 交易开始时间
        :type trade_start_time: pd.Timestamp
        :param trade_end_time: 交易结束时间
        :type trade_end_time: pd.Timestamp

        :rtype: list
        """
        raise NotImplementedError()


class OrderGenWInteract(OrderGenerator):
    """带交互功能的订单生成器"""

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        """generate_order_list_from_target_weight_position

        不对非流通股进行调整。所有可交易价值根据权重分配给可交易股票。
        如果interact == True，将使用交易日价格生成订单列表
        否则，仅使用交易日之前的价格生成订单列表

        :param current: 当前持仓
        :type current: Position
        :param trade_exchange: 交易所
        :type trade_exchange: Exchange
        :param target_weight_position: 目标权重持仓
        :type target_weight_position: dict
        :param risk_degree: 风险度
        :type risk_degree: float
        :param pred_start_time: 预测开始时间
        :type pred_start_time: pd.Timestamp
        :param pred_end_time: 预测结束时间
        :type pred_end_time: pd.Timestamp
        :param trade_start_time: 交易开始时间
        :type trade_start_time: pd.Timestamp
        :param trade_end_time: 交易结束时间
        :type trade_end_time: pd.Timestamp

        :rtype: list
        """
        if target_weight_position is None:
            return []

        # 计算当前可交易价值
        current_amount_dict = current.get_stock_amount_dict()

        current_total_value = trade_exchange.calculate_amount_position_value(
            amount_dict=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
            only_tradable=False,
        )
        current_tradable_value = trade_exchange.calculate_amount_position_value(
            amount_dict=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
            only_tradable=True,
        )
        # add cash
        current_tradable_value += current.get_cash()

        reserved_cash = (1.0 - risk_degree) * (current_total_value + current.get_cash())
        current_tradable_value -= reserved_cash

        if current_tradable_value < 0:
            # if you sell all the tradable stock can not meet the reserved
            # value. Then just sell all the stocks
            target_amount_dict = copy.deepcopy(current_amount_dict.copy())
            for stock_id in list(target_amount_dict.keys()):
                if trade_exchange.is_stock_tradable(stock_id, start_time=trade_start_time, end_time=trade_end_time):
                    del target_amount_dict[stock_id]
        else:
            # consider cost rate
            current_tradable_value /= 1 + max(trade_exchange.close_cost, trade_exchange.open_cost)

            # strategy 1 : generate amount_position by weight_position
            # Use API in Exchange()
            target_amount_dict = trade_exchange.generate_amount_position_from_weight_position(
                weight_position=target_weight_position,
                cash=current_tradable_value,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
        order_list = trade_exchange.generate_order_for_target_amount_position(
            target_position=target_amount_dict,
            current_position=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
        )
        return order_list


class OrderGenWOInteract(OrderGenerator):
    """Order Generator Without Interact"""

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        """generate_order_list_from_target_weight_position

        generate order list directly not using the information (e.g. whether can be traded, the accurate trade price)
         at trade date.
        In target weight position, generating order list need to know the price of objective stock in trade date,
        but we cannot get that
        value when do not interact with exchange, so we check the %close price at pred_date or price recorded
        in current position.

        :param current:
        :type current: Position
        :param trade_exchange:
        :type trade_exchange: Exchange
        :param target_weight_position:
        :type target_weight_position: dict
        :param risk_degree:
        :type risk_degree: float
        :param pred_start_time:
        :type pred_start_time: pd.Timestamp
        :param pred_end_time:
        :type pred_end_time: pd.Timestamp
        :param trade_start_time:
        :type trade_start_time: pd.Timestamp
        :param trade_end_time:
        :type trade_end_time: pd.Timestamp

        :rtype: list of generated orders
        """
        if target_weight_position is None:
            return []

        risk_total_value = risk_degree * current.calculate_value()

        current_stock = current.get_stock_list()
        amount_dict = {}
        for stock_id in target_weight_position:
            # Current rule will ignore the stock that not hold and cannot be traded at predict date
            if trade_exchange.is_stock_tradable(
                stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time
            ) and trade_exchange.is_stock_tradable(
                stock_id=stock_id, start_time=pred_start_time, end_time=pred_end_time
            ):
                amount_dict[stock_id] = (
                    risk_total_value
                    * target_weight_position[stock_id]
                    / trade_exchange.get_close(stock_id, start_time=pred_start_time, end_time=pred_end_time)
                )
                # TODO: Qlib use None to represent trading suspension.
                #  So last close price can't be the estimated trading price.
                # Maybe a close price with forward fill will be a better solution.
            elif stock_id in current_stock:
                amount_dict[stock_id] = (
                    risk_total_value * target_weight_position[stock_id] / current.get_stock_price(stock_id)
                )
            else:
                continue
        order_list = trade_exchange.generate_order_for_target_amount_position(
            target_position=amount_dict,
            current_position=current.get_stock_amount_dict(),
            start_time=trade_start_time,
            end_time=trade_end_time,
        )
        return order_list
