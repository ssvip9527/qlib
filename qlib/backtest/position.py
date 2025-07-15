# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from ..data.data import D
from .decision import Order


class BasePosition:
    """
    持仓类希望像字典一样维护持仓信息
    有关具体持仓实现，请参考`Position`类
    """

    def __init__(self, *args: Any, cash: float = 0.0, **kwargs: Any) -> None:
        self._settle_type = self.ST_NO
        self.position: dict = {}

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30) -> None:
        pass

    def skip_update(self) -> bool:
        """
        Should we skip updating operation for this position
        For example, updating is meaningless for InfPosition

        Returns
        -------
        bool:
            should we skip the updating operator
        """
        return False

    def check_stock(self, stock_id: str) -> bool:
        """
        检查股票是否在持仓中

        参数
        ----------
        stock_id : str
            股票ID

        返回
        -------
        bool:
            股票是否在持仓中的标志
        """
        raise NotImplementedError(f"Please implement the `check_stock` method")

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        """
        参数
        ----------
        order : Order
            用于更新持仓的订单
        trade_val : float
            交易结果的价值(金额)
        cost : float
            交易结果的成本
        trade_price : float
            交易执行价格
        """
        raise NotImplementedError(f"Please implement the `update_order` method")

    def update_stock_price(self, stock_id: str, price: float) -> None:
        """
        更新订单的最新价格
        在每个交易小节结束时清算余额时有用

        参数
        ----------
        stock_id :
            股票ID
        price : float
            要更新的价格
        """
        raise NotImplementedError(f"Please implement the `update_stock_price` method")

    def calculate_stock_value(self) -> float:
        """
        计算持仓中除现金外所有资产的价值

        返回
        -------
        float:
            所有股票的价值(金额)
        """
        raise NotImplementedError(f"Please implement the `calculate_stock_value` method")

    def calculate_value(self) -> float:
        raise NotImplementedError(f"Please implement the `calculate_value` method")

    def get_stock_list(self) -> List[str]:
        """
        获取持仓中的股票列表。
        """
        raise NotImplementedError(f"Please implement the `get_stock_list` method")

    def get_stock_price(self, code: str) -> float:
        """
        获取股票的最新价格

        参数
        ----------
        code :
            股票代码
        """
        raise NotImplementedError(f"Please implement the `get_stock_price` method")

    def get_stock_amount(self, code: str) -> float:
        """
        获取股票的数量

        参数
        ----------
        code :
            股票代码

        返回
        -------
        float:
            股票的数量
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount` method")

    def get_cash(self, include_settle: bool = False) -> float:
        """
        参数
        ----------
        include_settle:
            是否包含未结算(延迟)现金
            默认值: 不包含那些不可用现金

        返回
        -------
        float:
            持仓中的可用(可交易)现金
        """
        raise NotImplementedError(f"Please implement the `get_cash` method")

    def get_stock_amount_dict(self) -> dict:
        """
        生成股票数量字典 {stock_id : 股票数量}

        返回
        -------
        Dict:
            {stock_id : 股票数量}
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount_dict` method")

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """
        生成股票权重字典 {stock_id : 股票在持仓中的价值权重}
        在每个交易步骤的开始或结束时才有意义
        - 在每个交易步骤执行期间，权重可能与投资组合价值不一致

        参数
        ----------
        only_stock : bool
            如果only_stock=True，返回每只股票在股票总额中的权重
            如果only_stock=False，返回每只股票在总资产(股票+现金)中的权重

        返回
        -------
        Dict:
            {stock_id : 股票在持仓中的价值权重}
        """
        raise NotImplementedError(f"Please implement the `get_stock_weight_dict` method")

    def add_count_all(self, bar: str) -> None:
        """
        Will be called at the end of each bar on each level

        Parameters
        ----------
        bar :
            The level to be updated
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    def update_weight_all(self) -> None:
        """
        更新持仓权重；

        # TODO: 此函数有点奇怪。在处理订单后和更新权重前，持仓中的权重数据处于错误状态。
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    ST_CASH = "cash"
    ST_NO = "None"  # String is more typehint friendly than None

    def settle_start(self, settle_type: str) -> None:
        """
        结算开始
        其作用类似于开始并提交一笔交易

        参数
        ----------
        settle_type : str
            是否在每次执行中延迟结算（每次执行会使执行器前进一个步骤）
            - "cash": 延迟现金结算。
                - 获得的现金不能在当前步骤使用（例如，不能卖出股票获得现金来购买另一只股票）
            - None: 无结算机制
            - TODO: 未来将支持其他资产。
        """
        raise NotImplementedError(f"Please implement the `settle_conf` method")

    def settle_commit(self) -> None:
        """
        结算提交
        """
        raise NotImplementedError(f"Please implement the `settle_commit` method")

    def __str__(self) -> str:
        return self.__dict__.__str__()

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class Position(BasePosition):
    """Position

    current state of position
    a typical example is :{
      <instrument_id>: {
        'count': <how many days the security has been hold>,
        'amount': <the amount of the security>,
        'price': <the close price of security in the last trading day>,
        'weight': <the security weight of total position value>,
      },
    }
    """

    def __init__(self, cash: float = 0, position_dict: Dict[str, Union[Dict[str, float], float]] = {}) -> None:
        """Init position by cash and position_dict.

        Parameters
        ----------
        cash : float, optional
            initial cash in account, by default 0
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # it is equal to {"amount": int}
                                {"amount": int, "price"(optional): float},
                            ]
                        ]
            initial stocks with parameters amount and price,
            if there is no price key in the dict of stocks, it will be filled by _fill_stock_value.
            by default {}.
        """
        super().__init__()

        # NOTE: The position dict must be copied!!!
        # Otherwise the initial value
        self.init_cash = cash
        self.position = position_dict.copy()
        for stock, value in self.position.items():
            if isinstance(value, int):
                self.position[stock] = {"amount": value}
        self.position["cash"] = cash

        # If the stock price information is missing, the account value will not be calculated temporarily
        try:
            self.position["now_account_value"] = self.calculate_value()
        except KeyError:
            pass

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30) -> None:
        """fill the stock value by the close price of latest last_days from qlib.

        Parameters
        ----------
        start_time :
            the start time of backtest.
        freq : str
            Frequency
        last_days : int, optional
            the days to get the latest close price, by default 30.
        """
        stock_list = []
        for stock, value in self.position.items():
            if not isinstance(value, dict):
                continue
            if value.get("price", None) is None:
                stock_list.append(stock)

        if len(stock_list) == 0:
            return

        start_time = pd.Timestamp(start_time)
        # note that start time is 2020-01-01 00:00:00 if raw start time is "2020-01-01"
        price_end_time = start_time
        price_start_time = start_time - timedelta(days=last_days)
        price_df = D.features(
            stock_list,
            ["$close"],
            price_start_time,
            price_end_time,
            freq=freq,
            disk_cache=True,
        ).dropna()
        price_dict = price_df.groupby(["instrument"], group_keys=False).tail(1)["$close"].to_dict()

        if len(price_dict) < len(stock_list):
            lack_stock = set(stock_list) - set(price_dict)
            raise ValueError(f"{lack_stock} doesn't have close price in qlib in the latest {last_days} days")

        for stock in stock_list:
            self.position[stock]["price"] = price_dict[stock]
        self.position["now_account_value"] = self.calculate_value()

    def _init_stock(self, stock_id: str, amount: float, price: float | None = None) -> None:
        """
        initialization the stock in current position

        Parameters
        ----------
        stock_id :
            the id of the stock
        amount : float
            the amount of the stock
        price :
             the price when buying the init stock
        """
        self.position[stock_id] = {}
        self.position[stock_id]["amount"] = amount
        self.position[stock_id]["price"] = price
        self.position[stock_id]["weight"] = 0  # update the weight in the end of the trade date

    def _buy_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            self._init_stock(stock_id=stock_id, amount=trade_amount, price=trade_price)
        else:
            # exist, add amount
            self.position[stock_id]["amount"] += trade_amount

        self.position["cash"] -= trade_val + cost

    def _sell_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            raise KeyError("{} not in current position".format(stock_id))
        else:
            if np.isclose(self.position[stock_id]["amount"], trade_amount):
                # Selling all the stocks
                # we use np.isclose instead of abs(<the final amount>) <= 1e-5  because `np.isclose` consider both
                # relative amount and absolute amount
                # Using abs(<the final amount>) <= 1e-5 will result in error when the amount is large
                self._del_stock(stock_id)
            else:
                # decrease the amount of stock
                self.position[stock_id]["amount"] -= trade_amount
                # check if to delete
                if self.position[stock_id]["amount"] < -1e-5:
                    raise ValueError(
                        "only have {} {}, require {}".format(
                            self.position[stock_id]["amount"] + trade_amount,
                            stock_id,
                            trade_amount,
                        ),
                    )

        new_cash = trade_val - cost
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += new_cash
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def _del_stock(self, stock_id: str) -> None:
        del self.position[stock_id]

    def check_stock(self, stock_id: str) -> bool:
        return stock_id in self.position

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        # handle order, order is a order class, defined in exchange.py
        if order.direction == Order.BUY:
            # BUY
            self._buy_stock(order.stock_id, trade_val, cost, trade_price)
        elif order.direction == Order.SELL:
            # SELL
            self._sell_stock(order.stock_id, trade_val, cost, trade_price)
        else:
            raise NotImplementedError("do not support order direction {}".format(order.direction))

    def update_stock_price(self, stock_id: str, price: float) -> None:
        self.position[stock_id]["price"] = price

    def update_stock_count(self, stock_id: str, bar: str, count: float) -> None:  # TODO: check type of `bar`
        self.position[stock_id][f"count_{bar}"] = count

    def update_stock_weight(self, stock_id: str, weight: float) -> None:
        self.position[stock_id]["weight"] = weight

    def calculate_stock_value(self) -> float:
        stock_list = self.get_stock_list()
        value = 0
        for stock_id in stock_list:
            value += self.position[stock_id]["amount"] * self.position[stock_id]["price"]
        return value

    def calculate_value(self) -> float:
        value = self.calculate_stock_value()
        value += self.position["cash"] + self.position.get("cash_delay", 0.0)
        return value

    def get_stock_list(self) -> List[str]:
        stock_list = list(set(self.position.keys()) - {"cash", "now_account_value", "cash_delay"})
        return stock_list

    def get_stock_price(self, code: str) -> float:
        return self.position[code]["price"]

    def get_stock_amount(self, code: str) -> float:
        return self.position[code]["amount"] if code in self.position else 0

    def get_stock_count(self, code: str, bar: str) -> float:
        """the days the account has been hold, it may be used in some special strategies"""
        if f"count_{bar}" in self.position[code]:
            return self.position[code][f"count_{bar}"]
        else:
            return 0

    def get_stock_weight(self, code: str) -> float:
        return self.position[code]["weight"]

    def get_cash(self, include_settle: bool = False) -> float:
        cash = self.position["cash"]
        if include_settle:
            cash += self.position.get("cash_delay", 0.0)
        return cash

    def get_stock_amount_dict(self) -> dict:
        """generate stock amount dict {stock_id : amount of stock}"""
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.get_stock_amount(code=stock_code)
        return d

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """get_stock_weight_dict
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade date

        :param only_stock: If only_stock=True, the weight of each stock in total stock will be returned
                           If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned
        """
        if only_stock:
            position_value = self.calculate_stock_value()
        else:
            position_value = self.calculate_value()
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.position[stock_code]["amount"] * self.position[stock_code]["price"] / position_value
        return d

    def add_count_all(self, bar: str) -> None:
        stock_list = self.get_stock_list()
        for code in stock_list:
            if f"count_{bar}" in self.position[code]:
                self.position[code][f"count_{bar}"] += 1
            else:
                self.position[code][f"count_{bar}"] = 1

    def update_weight_all(self) -> None:
        weight_dict = self.get_stock_weight_dict()
        for stock_code, weight in weight_dict.items():
            self.update_stock_weight(stock_code, weight)

    def settle_start(self, settle_type: str) -> None:
        assert self._settle_type == self.ST_NO, "Currently, settlement can't be nested!!!!!"
        self._settle_type = settle_type
        if settle_type == self.ST_CASH:
            self.position["cash_delay"] = 0.0

    def settle_commit(self) -> None:
        if self._settle_type != self.ST_NO:
            if self._settle_type == self.ST_CASH:
                self.position["cash"] += self.position["cash_delay"]
                del self.position["cash_delay"]
            else:
                raise NotImplementedError(f"This type of input is not supported")
            self._settle_type = self.ST_NO


class InfPosition(BasePosition):
    """
    Position with infinite cash and amount.

    This is useful for generating random orders.
    """

    def skip_update(self) -> bool:
        """Updating state is meaningless for InfPosition"""
        return True

    def check_stock(self, stock_id: str) -> bool:
        # InfPosition always have any stocks
        return True

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        pass

    def update_stock_price(self, stock_id: str, price: float) -> None:
        pass

    def calculate_stock_value(self) -> float:
        """
        Returns
        -------
        float:
            infinity stock value
        """
        return np.inf

    def calculate_value(self) -> float:
        raise NotImplementedError(f"InfPosition doesn't support calculating value")

    def get_stock_list(self) -> List[str]:
        raise NotImplementedError(f"InfPosition doesn't support stock list position")

    def get_stock_price(self, code: str) -> float:
        """the price of the inf position is meaningless"""
        return np.nan

    def get_stock_amount(self, code: str) -> float:
        return np.inf

    def get_cash(self, include_settle: bool = False) -> float:
        return np.inf

    def get_stock_amount_dict(self) -> dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_amount_dict")

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_weight_dict")

    def add_count_all(self, bar: str) -> None:
        raise NotImplementedError(f"InfPosition doesn't support add_count_all")

    def update_weight_all(self) -> None:
        raise NotImplementedError(f"InfPosition doesn't support update_weight_all")

    def settle_start(self, settle_type: str) -> None:
        pass

    def settle_commit(self) -> None:
        pass
