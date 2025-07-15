# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union, cast

from ..utils.index_data import IndexData

if TYPE_CHECKING:
    from .account import Account

import random

import numpy as np
import pandas as pd

from qlib.backtest.position import BasePosition

from ..config import C
from ..constant import REG_CN, REG_TW
from ..data.data import D
from ..log import get_module_logger
from .decision import Order, OrderDir, OrderHelper
from .high_performance_ds import BaseQuote, NumpyQuote


class Exchange:
    # `quote_df`是一个包含回测基础信息的pd.DataFrame类
    # 经过处理后，数据将由`quote_cls`对象维护以实现更快的数据检索
    # `quote_df`的数据约定：
    # - $close用于计算每日结束时的总价值
    #   - 如果$close为None，则该日股票被视为停牌
    # - $factor用于交易单位的舍入
    #   - 当$close存在时，如果任何$factor缺失，将禁用交易单位舍入功能
    quote_df: pd.DataFrame

    def __init__(
        self,
        freq: str = "day",
        start_time: Union[pd.Timestamp, str] = None,
        end_time: Union[pd.Timestamp, str] = None,
        codes: Union[list, str] = "all",
        deal_price: Union[str, Tuple[str, str], List[str], None] = None,
        subscribe_fields: list = [],
        limit_threshold: Union[Tuple[str, str], float, None] = None,
        volume_threshold: Union[tuple, dict, None] = None,
        open_cost: float = 0.0015,
        close_cost: float = 0.0025,
        min_cost: float = 5.0,
        impact_cost: float = 0.0,
        extra_quote: pd.DataFrame = None,
        quote_cls: Type[BaseQuote] = NumpyQuote,
        **kwargs: Any,
    ) -> None:
        """
        初始化方法
        :param freq:             数据频率
        :param start_time:       回测的闭区间开始时间
        :param end_time:         回测的闭区间结束时间
        :param codes:            股票ID列表或工具字符串（如all, csi500, sse50）
        :param deal_price:      Union[str, Tuple[str, str], List[str]]
                                `deal_price`支持以下两种输入类型
                                - <deal_price> : str
                                - (<buy_price>, <sell_price>): 元组或列表
                                <deal_price>, <buy_price>或<sell_price> := <price>
                                <price> := str
                                - 例如'$close', '$open', '$vwap'（"close"也可以，`Exchange`会自动在表达式前添加"$"）
        :param subscribe_fields: 列表，订阅字段。这些表达式将添加到查询和`self.quote`中。
                                 当用户需要查询更多字段时非常有用
        :param limit_threshold: Union[Tuple[str, str], float, None]
                                1) `None`: 无限制
                                2) float，例如0.1，默认None
                                3) Tuple[str, str]: (<买入限制表达式>, <卖出限制表达式>)
                                                    `False`值表示股票可交易
                                                    `True`值表示股票受限不可交易
        :param volume_threshold: Union[
                                    Dict[
                                        "all": ("cum"或"current", 限制表达式),
                                        "buy": ("cum"或"current", 限制表达式),
                                        "sell":("cum"或"current", 限制表达式),
                                    ],
                                    ("cum"或"current", 限制表达式),
                                 ]
                                1) ("cum"或"current", 限制表达式)表示单一的成交量限制。
                                    - 限制表达式是qlib数据表达式，允许定义自己的操作符。
                                    请参考qlib/contrib/ops/high_freq.py，这里有高频数据的自定义操作符，如DayCumsum。
                                    !!!注意：如果要使用自定义操作符，需要在qlib_init中注册。
                                    - "cum"表示这是随时间累积的值，如累积成交量。因此当用作成交量限制时，需要减去已成交金额。
                                    - "current"表示实时值，不会随时间累积，因此可直接用作容量限制。
                                    例如：("cum", "0.2 * DayCumsum($volume, '9:45', '14:45')"), ("current", "$bidV1")
                                2) "all"表示买入和卖出均受此成交量限制。
                                "buy"表示买入的成交量限制。"sell"表示卖出的成交量限制。
                                不同的成交量限制将通过min()函数聚合。如果volume_threshold只是
                                ("cum"或"current", 限制表达式)而非字典，则默认适用于买入和卖出，即等同于{"all": ("cum"或"current", 限制表达式)}。
                                3) 示例："volume_threshold": {
                                            "all": ("cum", "0.2 * DayCumsum($volume, '9:45', '14:45')"),
                                            "buy": ("current", "$askV1"),
                                            "sell": ("current", "$bidV1"),
                                        }
        :param open_cost:        开仓成本率，默认0.0015
        :param close_cost:       平仓成本率，默认0.0025
        :param trade_unit:       交易单位，中国A股市场为100股。
                                 None表示禁用交易单位。
                                 **注意**：`trade_unit`包含在`kwargs`中。这是必要的，因为我们必须
                                 区分"未设置"和"禁用交易单位"
        :param min_cost:         最低成本，默认5
        :param impact_cost:     市场冲击成本率（也称为滑点）。推荐值为0.1。
        :param extra_quote:     pandas DataFrame，包含以下列：
                                    如['$vwap', '$close', '$volume', '$factor', 'limit_sell', 'limit_buy']。
                                    限制列表示ETF在特定日期是否可交易。
                                    必要字段：
                                        $close用于计算每日结束时的总价值。
                                    可选字段：
                                        $volume仅在限制交易金额或计算PA(vwap)指标时需要
                                        $vwap仅在使用$vwap价格作为成交价格时需要
                                        $factor用于舍入到交易单位
                                        limit_sell默认设为False（False表示当天可卖出该标的）。
                                        limit_buy默认设为False（False表示当天可买入该标的）。
                                    索引：MultiIndex(instrument, pd.Datetime)
        """
        self.freq = freq
        self.start_time = start_time
        self.end_time = end_time

        self.trade_unit = kwargs.pop("trade_unit", C.trade_unit)
        if len(kwargs) > 0:
            raise ValueError(f"Get Unexpected arguments {kwargs}")

        if limit_threshold is None:
            limit_threshold = C.limit_threshold
        if deal_price is None:
            deal_price = C.deal_price

        # we have some verbose information here. So logging is enabled
        self.logger = get_module_logger("online operator")

        # TODO: quote、trade_dates和codes不是必需的
        # 这些字段仅出于性能考虑而保留
        self.limit_type = self._get_limit_type(limit_threshold)
        if limit_threshold is None:
            if C.region in [REG_CN, REG_TW]:
                self.logger.warning(f"limit_threshold not set. The stocks hit the limit may be bought/sold")
        elif self.limit_type == self.LT_FLT and abs(cast(float, limit_threshold)) > 0.1:
            if C.region in [REG_CN, REG_TW]:
                self.logger.warning(f"limit_threshold may not be set to a reasonable value")

        if isinstance(deal_price, str):
            if deal_price[0] != "$":
                deal_price = "$" + deal_price
            self.buy_price = self.sell_price = deal_price
        elif isinstance(deal_price, (tuple, list)):
            self.buy_price, self.sell_price = cast(Tuple[str, str], deal_price)
        else:
            raise NotImplementedError(f"This type of input is not supported")

        if isinstance(codes, str):
            codes = D.instruments(codes)
        self.codes = codes
        # 必需字段
        # $close用于计算每日结束时的总价值
        # - 如果$close为None，则该日股票被视为停牌
        # $factor用于交易单位的舍入
        # $change用于计算股票的涨跌停限制

        # 从kwargs获取成交量限制
        self.buy_vol_limit, self.sell_vol_limit, vol_lt_fields = self._get_vol_limit(volume_threshold)

        necessary_fields = {self.buy_price, self.sell_price, "$close", "$change", "$factor", "$volume"}
        if self.limit_type == self.LT_TP_EXP:
            assert isinstance(limit_threshold, tuple)
            for exp in limit_threshold:
                necessary_fields.add(exp)
        all_fields = list(necessary_fields | set(vol_lt_fields) | set(subscribe_fields))

        self.all_fields = all_fields

        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost
        self.impact_cost = impact_cost

        self.limit_threshold: Union[Tuple[str, str], float, None] = limit_threshold
        self.volume_threshold = volume_threshold
        self.extra_quote = extra_quote
        self.get_quote_from_qlib()

        # init quote by quote_df
        self.quote_cls = quote_cls
        self.quote: BaseQuote = self.quote_cls(self.quote_df, freq)

    def get_quote_from_qlib(self) -> None:
        # get stock data from qlib
        if len(self.codes) == 0:
            self.codes = D.instruments()
        self.quote_df = D.features(
            self.codes,
            self.all_fields,
            self.start_time,
            self.end_time,
            freq=self.freq,
            disk_cache=True,
        )
        self.quote_df.columns = self.all_fields

        # check buy_price data and sell_price data
        for attr in ("buy_price", "sell_price"):
            pstr = getattr(self, attr)  # price string
            if self.quote_df[pstr].isna().any():
                self.logger.warning("{} field data contains nan.".format(pstr))

        # update trade_w_adj_price
        if (self.quote_df["$factor"].isna() & ~self.quote_df["$close"].isna()).any():
            # The 'factor.day.bin' file not exists, and `factor` field contains `nan`
            # Use adjusted price
            self.trade_w_adj_price = True
            self.logger.warning("factor.day.bin file not exists or factor contains `nan`. Order using adjusted_price.")
            if self.trade_unit is not None:
                self.logger.warning(f"trade unit {self.trade_unit} is not supported in adjusted_price mode.")
        else:
            # The `factor.day.bin` file exists and all data `close` and `factor` are not `nan`
            # Use normal price
            self.trade_w_adj_price = False
        # update limit
        self._update_limit(self.limit_threshold)

        # concat extra_quote
        if self.extra_quote is not None:
            # process extra_quote
            if "$close" not in self.extra_quote:
                raise ValueError("$close is necessray in extra_quote")
            for attr in "buy_price", "sell_price":
                pstr = getattr(self, attr)  # price string
                if pstr not in self.extra_quote.columns:
                    self.extra_quote[pstr] = self.extra_quote["$close"]
                    self.logger.warning(f"No {pstr} set for extra_quote. Use $close as {pstr}.")
            if "$factor" not in self.extra_quote.columns:
                self.extra_quote["$factor"] = 1.0
                self.logger.warning("No $factor set for extra_quote. Use 1.0 as $factor.")
            if "limit_sell" not in self.extra_quote.columns:
                self.extra_quote["limit_sell"] = False
                self.logger.warning("No limit_sell set for extra_quote. All stock will be able to be sold.")
            if "limit_buy" not in self.extra_quote.columns:
                self.extra_quote["limit_buy"] = False
                self.logger.warning("No limit_buy set for extra_quote. All stock will be able to be bought.")
            assert set(self.extra_quote.columns) == set(self.quote_df.columns) - {"$change"}
            self.quote_df = pd.concat([self.quote_df, self.extra_quote], sort=False, axis=0)

    LT_TP_EXP = "(exp)"  # Tuple[str, str]:  the limitation is calculated by a Qlib expression.
    LT_FLT = "float"  # float:  the trading limitation is based on `abs($change) < limit_threshold`
    LT_NONE = "none"  # none:  there is no trading limitation

    def _get_limit_type(self, limit_threshold: Union[tuple, float, None]) -> str:
        """get limit type"""
        if isinstance(limit_threshold, tuple):
            return self.LT_TP_EXP
        elif isinstance(limit_threshold, float):
            return self.LT_FLT
        elif limit_threshold is None:
            return self.LT_NONE
        else:
            raise NotImplementedError(f"This type of `limit_threshold` is not supported")

    def _update_limit(self, limit_threshold: Union[Tuple, float, None]) -> None:
        # $close may contain NaN, the nan indicates that the stock is not tradable at that timestamp
        suspended = self.quote_df["$close"].isna()
        # check limit_threshold
        limit_type = self._get_limit_type(limit_threshold)
        if limit_type == self.LT_NONE:
            self.quote_df["limit_buy"] = suspended
            self.quote_df["limit_sell"] = suspended
        elif limit_type == self.LT_TP_EXP:
            # set limit
            limit_threshold = cast(tuple, limit_threshold)
            # astype bool is necessary, because quote_df is an expression and could be float
            self.quote_df["limit_buy"] = self.quote_df[limit_threshold[0]].astype("bool") | suspended
            self.quote_df["limit_sell"] = self.quote_df[limit_threshold[1]].astype("bool") | suspended
        elif limit_type == self.LT_FLT:
            limit_threshold = cast(float, limit_threshold)
            self.quote_df["limit_buy"] = self.quote_df["$change"].ge(limit_threshold) | suspended
            self.quote_df["limit_sell"] = (
                self.quote_df["$change"].le(-limit_threshold) | suspended
            )  # pylint: disable=E1130

    @staticmethod
    def _get_vol_limit(volume_threshold: Union[tuple, dict, None]) -> Tuple[Optional[list], Optional[list], set]:
        """
        preprocess the volume limit.
        get the fields need to get from qlib.
        get the volume limit list of buying and selling which is composed of all limits.
        Parameters
        ----------
        volume_threshold :
            please refer to the doc of exchange.
        Returns
        -------
        fields: set
            the fields need to get from qlib.
        buy_vol_limit: List[Tuple[str]]
            all volume limits of buying.
        sell_vol_limit: List[Tuple[str]]
            all volume limits of selling.
        Raises
        ------
        ValueError
            the format of volume_threshold is not supported.
        """
        if volume_threshold is None:
            return None, None, set()

        fields = set()
        buy_vol_limit = []
        sell_vol_limit = []
        if isinstance(volume_threshold, tuple):
            volume_threshold = {"all": volume_threshold}

        assert isinstance(volume_threshold, dict)
        for key, vol_limit in volume_threshold.items():
            assert isinstance(vol_limit, tuple)
            fields.add(vol_limit[1])

            if key in ("buy", "all"):
                buy_vol_limit.append(vol_limit)
            if key in ("sell", "all"):
                sell_vol_limit.append(vol_limit)

        return buy_vol_limit, sell_vol_limit, fields

    def check_stock_limit(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: int | None = None,
    ) -> bool:
        """
        参数
        ----------
        stock_id : str
            股票ID
        start_time: pd.Timestamp
            开始时间
        end_time: pd.Timestamp
            结束时间
        direction : int, optional
            交易方向，默认为None
            - 如果direction为None，检查买入和卖出是否均可交易
            - 如果direction == Order.BUY，检查买入是否可交易
            - 如果direction == Order.SELL，检查卖出是否受限

        返回值
        -------
        True: 股票交易受限（可能达到最高价/最低价），因此股票不可交易
        False: 股票交易不受限，因此股票可交易
        """
        # NOTE:
        # **all** is used when checking limitation.
        # For example, the stock trading is limited in a day if every minute is limited in a day if every minute is limited.
        if direction is None:
            # The trading limitation is related to the trading direction
            # if the direction is not provided, then any limitation from buy or sell will result in trading limitation
            buy_limit = self.quote.get_data(stock_id, start_time, end_time, field="limit_buy", method="all")
            sell_limit = self.quote.get_data(stock_id, start_time, end_time, field="limit_sell", method="all")
            return bool(buy_limit or sell_limit)
        elif direction == Order.BUY:
            return cast(bool, self.quote.get_data(stock_id, start_time, end_time, field="limit_buy", method="all"))
        elif direction == Order.SELL:
            return cast(bool, self.quote.get_data(stock_id, start_time, end_time, field="limit_sell", method="all"))
        else:
            raise ValueError(f"direction {direction} is not supported!")

    def check_stock_suspended(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> bool:
        """如果股票停牌（因此不可交易），返回True"""
        # is suspended
        if stock_id in self.quote.get_all_stock():
            # suspended stocks are represented by None $close stock
            # The $close may contain NaN,
            close = self.quote.get_data(stock_id, start_time, end_time, "$close")
            if close is None:
                # if no close record exists
                return True
            elif isinstance(close, IndexData):
                # **any** non-NaN $close represents trading opportunity may exist
                #  if all returned is nan, then the stock is suspended
                return cast(bool, cast(IndexData, close).isna().all())
            else:
                # it is single value, make sure is not None
                return np.isnan(close)
        else:
            # if the stock is not in the stock list, then it is not tradable and regarded as suspended
            return True

    def is_stock_tradable(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: int | None = None,
    ) -> bool:
        # check if stock can be traded
        return not (
            self.check_stock_suspended(stock_id, start_time, end_time)
            or self.check_stock_limit(stock_id, start_time, end_time, direction)
        )

    def check_order(self, order: Order) -> bool:
        # check limit and suspended
        return self.is_stock_tradable(order.stock_id, order.start_time, order.end_time, order.direction)

    def deal_order(
        self,
        order: Order,
        trade_account: Account | None = None,
        position: BasePosition | None = None,
        dealt_order_amount: Dict[str, float] = defaultdict(float),
    ) -> Tuple[float, float, float]:
        """
        实际交易时处理订单
        `Order`中的结果部分将会被修改。
        :param order:  待处理的订单。
        :param trade_account: 处理订单后要更新的交易账户。
        :param position: 处理订单后要更新的持仓。
        :param dealt_order_amount: 已成交订单数量字典，格式为{stock_id: float}
        :return: trade_val（交易价值）, trade_cost（交易成本）, trade_price（交易价格）
        """
        # check order first.
        if not self.check_order(order):
            order.deal_amount = 0.0
            # using np.nan instead of None to make it more convenient to show the value in format string
            self.logger.debug(f"Order failed due to trading limitation: {order}")
            return 0.0, 0.0, np.nan

        if trade_account is not None and position is not None:
            raise ValueError("trade_account and position can only choose one")

        # NOTE: order will be changed in this function
        trade_price, trade_val, trade_cost = self._calc_trade_info_by_order(
            order,
            trade_account.current_position if trade_account else position,
            dealt_order_amount,
        )
        if trade_val > 1e-5:
            # If the order can only be deal 0 value. Nothing to be updated
            # Otherwise, it will result in
            # 1) some stock with 0 value in the position
            # 2) `trade_unit` of trade_cost will be lost in user account
            if trade_account:
                trade_account.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)
            elif position:
                position.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)

        return trade_val, trade_cost, trade_price

    def get_quote_info(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        field: str,
        method: str = "ts_data_last",
    ) -> Union[None, int, float, bool, IndexData]:
        return self.quote.get_data(stock_id, start_time, end_time, field=field, method=method)

    def get_close(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        method: str = "ts_data_last",
    ) -> Union[None, int, float, bool, IndexData]:
        return self.quote.get_data(stock_id, start_time, end_time, field="$close", method=method)

    def get_volume(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        method: Optional[str] = "sum",
    ) -> Union[None, int, float, bool, IndexData]:
        """获取股票`stock_id`在时间区间[start_time, end_time)内的总成交 volume"""
        return self.quote.get_data(stock_id, start_time, end_time, field="$volume", method=method)

    def get_deal_price(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: OrderDir,
        method: Optional[str] = "ts_data_last",
    ) -> Union[None, int, float, bool, IndexData]:
        if direction == OrderDir.SELL:
            pstr = self.sell_price
        elif direction == OrderDir.BUY:
            pstr = self.buy_price
        else:
            raise NotImplementedError(f"This type of input is not supported")

        deal_price = self.quote.get_data(stock_id, start_time, end_time, field=pstr, method=method)
        if method is not None and (deal_price is None or np.isnan(deal_price) or deal_price <= 1e-08):
            self.logger.warning(f"(stock_id:{stock_id}, trade_time:{(start_time, end_time)}, {pstr}): {deal_price}!!!")
            self.logger.warning(f"setting deal_price to close price")
            deal_price = self.get_close(stock_id, start_time, end_time, method)
        return deal_price

    def get_factor(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> Optional[float]:
        """
        Returns
        -------
        Optional[float]:
            `None`: if the stock is suspended `None` may be returned
            `float`: return factor if the factor exists
        """
        assert start_time is not None and end_time is not None, "the time range must be given"
        if stock_id not in self.quote.get_all_stock():
            return None
        return self.quote.get_data(stock_id, start_time, end_time, field="$factor", method="ts_data_last")

    def generate_amount_position_from_weight_position(
        self,
        weight_position: dict,
        cash: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: OrderDir = OrderDir.BUY,
    ) -> dict:
        """
        Generates the target position according to the weight and the cash.
        NOTE: All the cash will be assigned to the tradable stock.
        Parameter:
        weight_position : dict {stock_id : weight}; allocate cash by weight_position
            among then, weight must be in this range: 0 < weight < 1
        cash : cash
        start_time : the start time point of the step
        end_time : the end time point of the step
        direction : the direction of the deal price for estimating the amount
                    # NOTE: this function is used for calculating target position. So the default direction is buy
        """

        # calculate the total weight of tradable value
        tradable_weight = 0.0
        for stock_id, wp in weight_position.items():
            if self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):
                # weight_position must be greater than 0 and less than 1
                if wp < 0 or wp > 1:
                    raise ValueError(
                        "weight_position is {}, " "weight_position is not in the range of (0, 1).".format(wp),
                    )
                tradable_weight += wp

        if tradable_weight - 1.0 >= 1e-5:
            raise ValueError("tradable_weight is {}, can not greater than 1.".format(tradable_weight))

        amount_dict = {}
        for stock_id in weight_position:
            if weight_position[stock_id] > 0.0 and self.is_stock_tradable(
                stock_id=stock_id,
                start_time=start_time,
                end_time=end_time,
            ):
                amount_dict[stock_id] = (
                    cash
                    * weight_position[stock_id]
                    / tradable_weight
                    // self.get_deal_price(
                        stock_id=stock_id,
                        start_time=start_time,
                        end_time=end_time,
                        direction=direction,
                    )
                )
        return amount_dict

    def get_real_deal_amount(self, current_amount: float, target_amount: float, factor: float | None = None) -> float:
        """
        Calculate the real adjust deal amount when considering the trading unit
        :param current_amount:
        :param target_amount:
        :param factor:
        :return  real_deal_amount;  Positive deal_amount indicates buying more stock.
        """
        if current_amount == target_amount:
            return 0
        elif current_amount < target_amount:
            deal_amount = target_amount - current_amount
            deal_amount = self.round_amount_by_trade_unit(deal_amount, factor)
            return deal_amount
        else:
            if target_amount == 0:
                return -current_amount
            else:
                deal_amount = current_amount - target_amount
                deal_amount = self.round_amount_by_trade_unit(deal_amount, factor)
                return -deal_amount

    def generate_order_for_target_amount_position(
        self,
        target_position: dict,
        current_position: dict,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> List[Order]:
        """
        Note: some future information is used in this function
        Parameter:
        target_position : dict { stock_id : amount }
        current_position : dict { stock_id : amount}
        trade_unit : trade_unit
        down sample : for amount 321 and trade_unit 100, deal_amount is 300
        deal order on trade_date
        """
        # split buy and sell for further use
        buy_order_list = []
        sell_order_list = []
        # three parts: kept stock_id, dropped stock_id, new stock_id
        # handle kept stock_id

        # because the order of the set is not fixed, the trading order of the stock is different, so that the backtest
        # results of the same parameter are different;
        # so here we sort stock_id, and then randomly shuffle the order of stock_id
        # because the same random seed is used, the final stock_id order is fixed
        sorted_ids = sorted(set(list(current_position.keys()) + list(target_position.keys())))
        random.seed(0)
        random.shuffle(sorted_ids)
        for stock_id in sorted_ids:
            # Do not generate order for the non-tradable stocks
            if not self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):
                continue

            target_amount = target_position.get(stock_id, 0)
            current_amount = current_position.get(stock_id, 0)
            factor = self.get_factor(stock_id, start_time=start_time, end_time=end_time)

            deal_amount = self.get_real_deal_amount(current_amount, target_amount, factor)
            if deal_amount == 0:
                continue
            if deal_amount > 0:
                # buy stock
                buy_order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=deal_amount,
                        direction=Order.BUY,
                        start_time=start_time,
                        end_time=end_time,
                        factor=factor,
                    ),
                )
            else:
                # sell stock
                sell_order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=abs(deal_amount),
                        direction=Order.SELL,
                        start_time=start_time,
                        end_time=end_time,
                        factor=factor,
                    ),
                )
        # return order_list : buy + sell
        return sell_order_list + buy_order_list

    def calculate_amount_position_value(
        self,
        amount_dict: dict,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        only_tradable: bool = False,
        direction: OrderDir = OrderDir.SELL,
    ) -> float:
        """Parameter
        position : Position()
        amount_dict : {stock_id : amount}
        direction : the direction of the deal price for estimating the amount
                    # NOTE:
                    This function is used for calculating current position value.
                    So the default direction is sell.
        """
        value = 0
        for stock_id in amount_dict:
            if not only_tradable or (
                not self.check_stock_suspended(stock_id=stock_id, start_time=start_time, end_time=end_time)
                and not self.check_stock_limit(stock_id=stock_id, start_time=start_time, end_time=end_time)
            ):
                value += (
                    self.get_deal_price(
                        stock_id=stock_id,
                        start_time=start_time,
                        end_time=end_time,
                        direction=direction,
                    )
                    * amount_dict[stock_id]
                )
        return value

    def _get_factor_or_raise_error(
        self,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> float:
        """Please refer to the docs of get_amount_of_trade_unit"""
        if factor is None:
            if stock_id is not None and start_time is not None and end_time is not None:
                factor = self.get_factor(stock_id=stock_id, start_time=start_time, end_time=end_time)
            else:
                raise ValueError(f"`factor` and (`stock_id`, `start_time`, `end_time`) can't both be None")
        assert factor is not None
        return factor

    def get_amount_of_trade_unit(
        self,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> Optional[float]:
        """
        get the trade unit of amount based on **factor**
        the factor can be given directly or calculated in given time range and stock id.
        `factor` has higher priority than `stock_id`, `start_time` and `end_time`
        Parameters
        ----------
        factor : float
            the adjusted factor
        stock_id : str
            the id of the stock
        start_time :
            the start time of trading range
        end_time :
            the end time of trading range
        """
        if not self.trade_w_adj_price and self.trade_unit is not None:
            factor = self._get_factor_or_raise_error(
                factor=factor,
                stock_id=stock_id,
                start_time=start_time,
                end_time=end_time,
            )
            return self.trade_unit / factor
        else:
            return None

    def round_amount_by_trade_unit(
        self,
        deal_amount: float,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> float:
        """Parameter
        Please refer to the docs of get_amount_of_trade_unit
        deal_amount : float, adjusted amount
        factor : float, adjusted factor
        return : float, real amount
        """
        if not self.trade_w_adj_price and self.trade_unit is not None:
            # the minimal amount is 1. Add 0.1 for solving precision problem.
            factor = self._get_factor_or_raise_error(
                factor=factor,
                stock_id=stock_id,
                start_time=start_time,
                end_time=end_time,
            )
            return (deal_amount * factor + 0.1) // self.trade_unit * self.trade_unit / factor
        return deal_amount

    def _clip_amount_by_volume(self, order: Order, dealt_order_amount: dict) -> Optional[float]:
        """parse the capacity limit string and return the actual amount of orders that can be executed.
        NOTE:
            this function will change the order.deal_amount **inplace**
            - This will make the order info more accurate
        Parameters
        ----------
        order : Order
            the order to be executed.
        dealt_order_amount : dict
            :param dealt_order_amount: the dealt order amount dict with the format of {stock_id: float}
        """
        vol_limit = self.buy_vol_limit if order.direction == Order.BUY else self.sell_vol_limit

        if vol_limit is None:
            return order.deal_amount

        vol_limit_num: List[float] = []
        for limit in vol_limit:
            assert isinstance(limit, tuple)
            if limit[0] == "current":
                limit_value = self.quote.get_data(
                    order.stock_id,
                    order.start_time,
                    order.end_time,
                    field=limit[1],
                    method="sum",
                )
                vol_limit_num.append(cast(float, limit_value))
            elif limit[0] == "cum":
                limit_value = self.quote.get_data(
                    order.stock_id,
                    order.start_time,
                    order.end_time,
                    field=limit[1],
                    method="ts_data_last",
                )
                vol_limit_num.append(limit_value - dealt_order_amount[order.stock_id])
            else:
                raise ValueError(f"{limit[0]} is not supported")
        vol_limit_min = min(vol_limit_num)
        orig_deal_amount = order.deal_amount
        order.deal_amount = max(min(vol_limit_min, orig_deal_amount), 0)
        if vol_limit_min < orig_deal_amount:
            self.logger.debug(f"Order clipped due to volume limitation: {order}, {list(zip(vol_limit_num, vol_limit))}")

        return None

    def _get_buy_amount_by_cash_limit(self, trade_price: float, cash: float, cost_ratio: float) -> float:
        """return the real order amount after cash limit for buying.
        Parameters
        ----------
        trade_price : float
        cash : float
        cost_ratio : float

        Return
        ----------
        float
            the real order amount after cash limit for buying.
        """
        max_trade_amount = 0.0
        if cash >= self.min_cost:
            # critical_price means the stock transaction price when the service fee is equal to min_cost.
            critical_price = self.min_cost / cost_ratio + self.min_cost
            if cash >= critical_price:
                # the service fee is equal to cost_ratio * trade_amount
                max_trade_amount = cash / (1 + cost_ratio) / trade_price
            else:
                # the service fee is equal to min_cost
                max_trade_amount = (cash - self.min_cost) / trade_price
        return max_trade_amount

    def _calc_trade_info_by_order(
        self,
        order: Order,
        position: Optional[BasePosition],
        dealt_order_amount: dict,
    ) -> Tuple[float, float, float]:
        """
        Calculation of trade info
        **NOTE**: Order will be changed in this function
        :param order:
        :param position: Position
        :param dealt_order_amount: the dealt order amount dict with the format of {stock_id: float}
        :return: trade_price, trade_val, trade_cost
        """
        trade_price = cast(
            float,
            self.get_deal_price(order.stock_id, order.start_time, order.end_time, direction=order.direction),
        )
        total_trade_val = cast(float, self.get_volume(order.stock_id, order.start_time, order.end_time)) * trade_price
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        order.deal_amount = order.amount  # set to full amount and clip it step by step
        # Clipping amount first
        # - It simulates that the order is rejected directly by the exchange due to large order
        # Another choice is placing it after rounding the order
        # - It simulates that the large order is submitted, but partial is dealt regardless of rounding by trading unit.
        self._clip_amount_by_volume(order, dealt_order_amount)

        # TODO: the adjusted cost ratio can be overestimated as deal_amount will be clipped in the next steps
        trade_val = order.deal_amount * trade_price
        if not total_trade_val or np.isnan(total_trade_val):
            # TODO: assert trade_val == 0, f"trade_val != 0, total_trade_val: {total_trade_val}; order info: {order}"
            adj_cost_ratio = self.impact_cost
        else:
            adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2

        if order.direction == Order.SELL:
            cost_ratio = self.close_cost + adj_cost_ratio
            # sell
            # if we don't know current position, we choose to sell all
            # Otherwise, we clip the amount based on current position
            if position is not None:
                current_amount = (
                    position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
                )
                if not np.isclose(order.deal_amount, current_amount):
                    # when not selling last stock. rounding is necessary
                    order.deal_amount = self.round_amount_by_trade_unit(
                        min(current_amount, order.deal_amount),
                        order.factor,
                    )

                # in case of negative value of cash
                if position.get_cash() + order.deal_amount * trade_price < max(
                    order.deal_amount * trade_price * cost_ratio,
                    self.min_cost,
                ):
                    order.deal_amount = 0
                    self.logger.debug(f"Order clipped due to cash limitation: {order}")

        elif order.direction == Order.BUY:
            cost_ratio = self.open_cost + adj_cost_ratio
            # buy
            if position is not None:
                cash = position.get_cash()
                trade_val = order.deal_amount * trade_price
                if cash < max(trade_val * cost_ratio, self.min_cost):
                    # cash cannot cover cost
                    order.deal_amount = 0
                    self.logger.debug(f"Order clipped due to cost higher than cash: {order}")
                elif cash < trade_val + max(trade_val * cost_ratio, self.min_cost):
                    # The money is not enough
                    max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, cost_ratio)
                    order.deal_amount = self.round_amount_by_trade_unit(
                        min(max_buy_amount, order.deal_amount),
                        order.factor,
                    )
                    self.logger.debug(f"Order clipped due to cash limitation: {order}")
                else:
                    # The money is enough
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
            else:
                # Unknown amount of money. Just round the amount
                order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)

        else:
            raise NotImplementedError("order direction {} error".format(order.direction))

        trade_val = order.deal_amount * trade_price
        trade_cost = max(trade_val * cost_ratio, self.min_cost)
        if trade_val <= 1e-5:
            # if dealing is not successful, the trade_cost should be zero.
            trade_cost = 0
        return trade_price, trade_val, trade_cost

    def get_order_helper(self) -> OrderHelper:
        if not hasattr(self, "_order_helper"):
            # cache to avoid recreate the same instance
            self._order_helper = OrderHelper(self)
        return self._order_helper
