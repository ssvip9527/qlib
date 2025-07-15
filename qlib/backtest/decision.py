# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from abc import abstractmethod
from datetime import time
from enum import IntEnum

# try to fix circular imports when enabling type hints
from typing import TYPE_CHECKING, Any, ClassVar, Generic, List, Optional, Tuple, TypeVar, Union, cast

from qlib.backtest.utils import TradeCalendarManager
from qlib.data.data import Cal
from qlib.log import get_module_logger
from qlib.utils.time import concat_date_time, epsilon_change

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.exchange import Exchange

from dataclasses import dataclass

import numpy as np
import pandas as pd

DecisionType = TypeVar("DecisionType")


class OrderDir(IntEnum):
    # Order direction
    SELL = 0
    BUY = 1


@dataclass
class Order:
    """
    stock_id : str
        股票ID
    amount : float
        交易数量（非负且已调整的值）
    start_time : pd.Timestamp
        订单交易的闭区间开始时间
    end_time : pd.Timestamp
        订单交易的闭区间结束时间
    direction : int
        Order.SELL表示卖出；Order.BUY表示买入
    factor : float
        表示在Exchange()中分配的权重因子
    """

    # 1) time invariant values
    # - 这些字段由用户设置且不随时间变化
    stock_id: str
    amount: float  # `amount`是一个非负的已调整值
    direction: OrderDir

    # 2) 随时间变化的字段:
    # - 用户在使用底层API时可能需要设置这些值
    # - 如果用户不设置，TradeDecisionWO会帮助用户设置
    # 订单所属的时间区间(注意：这不是预期的订单执行时间范围)
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    # 3) 结果字段
    # - 用户通常不需要关心这些值
    # - 它们由回测系统在执行完成后设置
    # 各种情况下这些值应该是什么
    # - 不可交易: deal_amount == 0, factor为None
    #    - 股票停牌，整个订单失败。该订单无成本
    # - 已成交或部分成交: deal_amount >= 0 且 factor不为None
    deal_amount: float = 0.0  # `deal_amount`是一个非负值
    factor: Optional[float] = None

    # TODO:
    # 添加一个状态字段来指示订单的执行结果

    # FIXME:
    # 目前为了兼容性保留
    # 将来请移除这些字段
    SELL: ClassVar[OrderDir] = OrderDir.SELL
    BUY: ClassVar[OrderDir] = OrderDir.BUY

    def __post_init__(self) -> None:
        if self.direction not in {Order.SELL, Order.BUY}:
            raise NotImplementedError("direction not supported, `Order.SELL` for sell, `Order.BUY` for buy")
        self.deal_amount = 0.0
        self.factor = None

    @property
    def amount_delta(self) -> float:
        """
        返回amount的差值
        - 正值表示买入`amount`数量的股票
        - 负值表示卖出`amount`数量的股票
        """
        return self.amount * self.sign

    @property
    def deal_amount_delta(self) -> float:
        """
        返回deal_amount的差值
        - 正值表示买入`deal_amount`数量的股票
        - 负值表示卖出`deal_amount`数量的股票
        """
        return self.deal_amount * self.sign

    @property
    def sign(self) -> int:
        """
        返回交易方向符号
        - `+1`表示买入
        - `-1`表示卖出
        """
        return self.direction * 2 - 1

    @staticmethod
    def parse_dir(direction: Union[str, int, np.integer, OrderDir, np.ndarray]) -> Union[OrderDir, np.ndarray]:
        if isinstance(direction, OrderDir):
            return direction
        elif isinstance(direction, (int, float, np.integer, np.floating)):
            return Order.BUY if direction > 0 else Order.SELL
        elif isinstance(direction, str):
            dl = direction.lower().strip()
            if dl == "sell":
                return OrderDir.SELL
            elif dl == "buy":
                return OrderDir.BUY
            else:
                raise NotImplementedError(f"This type of input is not supported")
        elif isinstance(direction, np.ndarray):
            direction_array = direction.copy()
            direction_array[direction_array > 0] = Order.BUY
            direction_array[direction_array <= 0] = Order.SELL
            return direction_array
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @property
    def key_by_day(self) -> tuple:
        """返回一个可哈希且唯一的键，用于在日粒度上标识该订单"""
        return self.stock_id, self.date, self.direction

    @property
    def key(self) -> tuple:
        """返回一个可哈希且唯一的键，用于标识该订单"""
        return self.stock_id, self.start_time, self.end_time, self.direction

    @property
    def date(self) -> pd.Timestamp:
        """返回订单的日期"""
        return pd.Timestamp(self.start_time.replace(hour=0, minute=0, second=0))


class OrderHelper:
    """
    设计目的
    - 简化订单生成过程
        - 用户可能不了解系统中的调整因子信息
        - 生成订单时需要与交易所进行过多交互
    """

    def __init__(self, exchange: Exchange) -> None:
        self.exchange = exchange

    @staticmethod
    def create(
        code: str,
        amount: float,
        direction: OrderDir,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
    ) -> Order:
        """
        帮助创建订单

        # TODO: create order for unadjusted amount order

        参数
        ----------
        code : str
            标的证券ID
        amount : float
            **已调整的交易数量**
        direction : OrderDir
            交易方向
        start_time : Union[str, pd.Timestamp] (可选)
            订单所属的时间区间开始时间
        end_time : Union[str, pd.Timestamp] (可选)
            订单所属的时间区间结束时间

        返回值
        -------
        Order:
            创建的订单对象
        """
        # NOTE: factor is a value belongs to the results section. User don't have to care about it when creating orders
        return Order(
            stock_id=code,
            amount=amount,
            start_time=None if start_time is None else pd.Timestamp(start_time),
            end_time=None if end_time is None else pd.Timestamp(end_time),
            direction=direction,
        )


class TradeRange:
    @abstractmethod
    def __call__(self, trade_calendar: TradeCalendarManager) -> Tuple[int, int]:
        """
        此方法将按以下方式调用

        外部策略通过`TradeRange`给出决策
        该决策将由内部决策进行检查
        内部决策在获取交易范围时会将其trade_calendar作为参数传入
        - 框架的步骤基于整数索引

        参数
        ----------
        trade_calendar : TradeCalendarManager
            来自内部策略的交易日历

        返回值
        -------
        Tuple[int, int]:
            可交易的开始索引和结束索引

        异常
        ------
        NotImplementedError:
            当没有范围限制时引发异常
        """
        raise NotImplementedError(f"Please implement the `__call__` method")

    @abstractmethod
    def clip_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        参数
        ----------
        start_time : pd.Timestamp
            开始时间
        end_time : pd.Timestamp
            结束时间（start_time和end_time均为闭区间）

        返回值
        -------
        Tuple[pd.Timestamp, pd.Timestamp]:
            可交易的时间范围。
            - 即[start_time, end_time]与TradeRange自身规则的交集
        """
        raise NotImplementedError(f"Please implement the `clip_time_range` method")


class IdxTradeRange(TradeRange):
    def __init__(self, start_idx: int, end_idx: int) -> None:
        self._start_idx = start_idx
        self._end_idx = end_idx

    def __call__(self, trade_calendar: TradeCalendarManager | None = None) -> Tuple[int, int]:
        return self._start_idx, self._end_idx

    def clip_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        raise NotImplementedError


class TradeRangeByTime(TradeRange):
    """This is a helper function for make decisions"""

    def __init__(self, start_time: str | time, end_time: str | time) -> None:
        """
        这是一个可调用类。

        **注意**:
        - 专为日内交易的分钟级K线设计！！！
        - start_time和end_time在范围内均为**闭区间**

        参数
        ----------
        start_time : str | time
            例如："9:30"
        end_time : str | time
            例如："14:30"
        """
        self.start_time = pd.Timestamp(start_time).time() if isinstance(start_time, str) else start_time
        self.end_time = pd.Timestamp(end_time).time() if isinstance(end_time, str) else end_time
        assert self.start_time < self.end_time

    def __call__(self, trade_calendar: TradeCalendarManager) -> Tuple[int, int]:
        if trade_calendar is None:
            raise NotImplementedError("trade_calendar is necessary for getting TradeRangeByTime.")

        start_date = trade_calendar.start_time.date()
        val_start, val_end = concat_date_time(start_date, self.start_time), concat_date_time(start_date, self.end_time)
        return trade_calendar.get_range_idx(val_start, val_end)

    def clip_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_date = start_time.date()
        val_start, val_end = concat_date_time(start_date, self.start_time), concat_date_time(start_date, self.end_time)
        # NOTE: `end_date` should not be used. Because the `end_date` is for slicing. It may be in the next day
        # Assumption: start_time and end_time is for intra-day trading. So it is OK for only using start_date
        return max(val_start, start_time), min(val_end, end_time)


class BaseTradeDecision(Generic[DecisionType]):
    """
    Trade decisions are made by strategy and executed by executor

    Motivation:
        Here are several typical scenarios for `BaseTradeDecision`

        Case 1:
        1. Outer strategy makes a decision. The decision is not available at the start of current interval
        2. After a period of time, the decision are updated and become available
        3. The inner strategy try to get the decision and start to execute the decision according to `get_range_limit`
        Case 2:
        1. The outer strategy's decision is available at the start of the interval
        2. Same as `case 1.3`
    """

    def __init__(self, strategy: BaseStrategy, trade_range: Union[Tuple[int, int], TradeRange, None] = None) -> None:
        """
        参数
        ----------
        strategy : BaseStrategy
            做出决策的策略实例
        trade_range: Union[Tuple[int, int], Callable] (可选)
            底层策略的索引范围。

            以下是每种类型的trade_range示例：

            1) Tuple[int, int]
            底层策略的开始索引和结束索引（均为闭区间）

            2) TradeRange

        """
        self.strategy = strategy
        self.start_time, self.end_time = strategy.trade_calendar.get_step_time()
        # upper strategy has no knowledge about the sub executor before `_init_sub_trading`
        self.total_step: Optional[int] = None
        if isinstance(trade_range, tuple):
            # for Tuple[int, int]
            trade_range = IdxTradeRange(*trade_range)
        self.trade_range: Optional[TradeRange] = trade_range

    def get_decision(self) -> List[DecisionType]:
        """
        获取**具体决策**（例如执行订单）
        此方法将由内部策略调用

        返回值
        -------
        List[DecisionType:
            决策结果。通常是一些订单
            示例:
                []:
                    决策不可用
                [concrete_decision]:
                    决策可用
        """
        raise NotImplementedError(f"This type of input is not supported")

    def update(self, trade_calendar: TradeCalendarManager) -> Optional[BaseTradeDecision]:
        """
        在每个步骤的**开始**时被调用。

        此函数设计用于以下目的：
        1) 为做出`self`决策的策略留下更新决策本身的钩子
        2) 从内部执行器日历更新一些信息

        参数
        ----------
        trade_calendar : TradeCalendarManager
            **内部策略**的日历！！！

        返回值
        -------
        BaseTradeDecision:
            新的更新，使用新决策。如果没有更新，返回None（使用先前的决策或不可用状态）
        """
        # purpose 1)
        self.total_step = trade_calendar.get_trade_len()

        # purpose 2)
        return self.strategy.update_trade_decision(self, trade_calendar)

    def _get_range_limit(self, **kwargs: Any) -> Tuple[int, int]:
        if self.trade_range is not None:
            return self.trade_range(trade_calendar=cast(TradeCalendarManager, kwargs.get("inner_calendar")))
        else:
            raise NotImplementedError("The decision didn't provide an index range")

    def get_range_limit(self, **kwargs: Any) -> Tuple[int, int]:
        """
        返回用于限制决策执行时间的预期步骤范围
        左右边界均为**闭区间**

        如果没有可用的trade_range，将返回`default_value`

        此方法仅在`NestedExecutor`中使用
        - 最外层策略不遵循任何范围限制（但可能提供range_limit）
        - 最内层策略的range_limit无效，因为原子执行器没有此类功能

        **注意**:
        1) 在以下情况下，此函数必须在`self.update`之后调用（由NestedExecutor确保）：
        - 用户依赖`self.update`的自动裁剪功能

        2) 此函数将在NestedExecutor的_init_sub_trading之后调用

        参数
        ----------
        **kwargs:
            {
                "default_value": <default_value>, # 使用字典是为了区分未提供值和提供None值的情况
                "inner_calendar": <内部策略的交易日历>
                # 因为范围限制将控制内部策略的步骤范围，当trade_range为可调用对象时，内部日历是重要参数
            }

        返回值
        -------
        Tuple[int, int]:
            步骤范围的开始索引和结束索引

        异常
        ------
        NotImplementedError:
            当满足以下条件时引发：
            1) 决策无法提供统一的开始和结束索引
            2) 未提供default_value
        """
        try:
            _start_idx, _end_idx = self._get_range_limit(**kwargs)
        except NotImplementedError as e:
            if "default_value" in kwargs:
                return kwargs["default_value"]
            else:
                # Default to get full index
                raise NotImplementedError(f"The decision didn't provide an index range") from e

        # clip index
        if getattr(self, "total_step", None) is not None:
            # if `self.update` is called.
            # Then the _start_idx, _end_idx should be clipped
            assert self.total_step is not None
            if _start_idx < 0 or _end_idx >= self.total_step:
                logger = get_module_logger("decision")
                logger.warning(
                    f"[{_start_idx},{_end_idx}] go beyond the total_step({self.total_step}), it will be clipped.",
                )
                _start_idx, _end_idx = max(0, _start_idx), min(self.total_step - 1, _end_idx)
        return _start_idx, _end_idx

    def get_data_cal_range_limit(self, rtype: str = "full", raise_error: bool = False) -> Tuple[int, int]:
        """
        根据数据日历获取范围限制

        注意：这是**整体**范围限制，而非单个步骤

        基于以下假设：
        1) common_infra中交易所的频率与数据日历相同
        2) 用户希望按**天**（即240分钟）对索引取模

        参数
        ----------
        rtype: str
            - "full": 返回当日决策的完整限制范围
            - "step": 返回当前步骤的限制范围

        raise_error: bool
            True: 如果未设置trade_range则引发错误
            False: 返回完整的交易日历

            在以下情况下很有用：
            - 当决策级别的交易范围不可用时，用户希望遵循订单特定的交易时间范围。
              引发NotImplementedError表示范围限制不可用

        返回值
        -------
        Tuple[int, int]:
            数据日历中的范围限制

        异常
        ------
        NotImplementedError:
            当满足以下条件时引发：
            1) 决策无法提供统一的开始和结束
            2) raise_error为True
        """
        # potential performance issue
        day_start = pd.Timestamp(self.start_time.date())
        day_end = epsilon_change(day_start + pd.Timedelta(days=1))
        freq = self.strategy.trade_exchange.freq
        _, _, day_start_idx, day_end_idx = Cal.locate_index(day_start, day_end, freq=freq)
        if self.trade_range is None:
            if raise_error:
                raise NotImplementedError(f"There is no trade_range in this case")
            else:
                return 0, day_end_idx - day_start_idx
        else:
            if rtype == "full":
                val_start, val_end = self.trade_range.clip_time_range(day_start, day_end)
            elif rtype == "step":
                val_start, val_end = self.trade_range.clip_time_range(self.start_time, self.end_time)
            else:
                raise ValueError(f"This type of input {rtype} is not supported")
            _, _, start_idx, end_index = Cal.locate_index(val_start, val_end, freq=freq)
            return start_idx - day_start_idx, end_index - day_start_idx

    def empty(self) -> bool:
        for obj in self.get_decision():
            if isinstance(obj, Order):
                # Zero amount order will be treated as empty
                if obj.amount > 1e-6:
                    return False
            else:
                return True
        return True

    def mod_inner_decision(self, inner_trade_decision: BaseTradeDecision) -> None:
        """
        此方法将在inner_trade_decision生成后被调用。
        `inner_trade_decision`将被**就地**修改。

        `mod_inner_decision`的设计动机：
        - 为外部决策留下影响内部策略生成的决策的钩子
            - 例如：最外层策略生成交易时间范围。但在原始设计中，上层只能影响最近的一层。
              通过`mod_inner_decision`，决策可以传递多个层级

        参数
        ----------
        inner_trade_decision : BaseTradeDecision
            内部交易决策实例
        """
        # base class provide a default behaviour to modify inner_trade_decision
        # trade_range should be propagated when inner trade_range is not set
        if inner_trade_decision.trade_range is None:
            inner_trade_decision.trade_range = self.trade_range


class EmptyTradeDecision(BaseTradeDecision[object]):
    def get_decision(self) -> List[object]:
        return []

    def empty(self) -> bool:
        return True


class TradeDecisionWO(BaseTradeDecision[Order]):
    """
    交易决策（包含订单）。
    此外，还包括时间范围。
    """

    def __init__(
        self,
        order_list: List[Order],
        strategy: BaseStrategy,
        trade_range: Union[Tuple[int, int], TradeRange, None] = None,
    ) -> None:
        super().__init__(strategy, trade_range=trade_range)
        self.order_list = cast(List[Order], order_list)
        start, end = strategy.trade_calendar.get_step_time()
        for o in order_list:
            assert isinstance(o, Order)
            if o.start_time is None:
                o.start_time = start
            if o.end_time is None:
                o.end_time = end

    def get_decision(self) -> List[Order]:
        return self.order_list

    def __repr__(self) -> str:
        return (
            f"class: {self.__class__.__name__}; "
            f"strategy: {self.strategy}; "
            f"trade_range: {self.trade_range}; "
            f"order_list[{len(self.order_list)}]"
        )


class TradeDecisionWithDetails(TradeDecisionWO):
    """
    包含详细信息的决策。
    详细信息用于生成执行报告。
    """

    def __init__(
        self,
        order_list: List[Order],
        strategy: BaseStrategy,
        trade_range: Optional[Tuple[int, int]] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(order_list, strategy, trade_range)

        self.details = details
