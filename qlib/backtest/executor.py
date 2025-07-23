from __future__ import annotations

import copy
from abc import abstractmethod
from collections import defaultdict
from types import GeneratorType
from typing import Any, Dict, Generator, List, Tuple, Union, cast

import pandas as pd

from qlib.backtest.account import Account
from qlib.backtest.position import BasePosition
from qlib.log import get_module_logger

from ..strategy.base import BaseStrategy
from ..utils import init_instance_by_config
from .decision import BaseTradeDecision, Order
from .exchange import Exchange
from .utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager, get_start_end_idx


class BaseExecutor:
    """交易的基础执行器"""

    def __init__(
        self,
        time_per_step: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        trade_exchange: Exchange | None = None,
        common_infra: CommonInfrastructure | None = None,
        settle_type: str = BasePosition.ST_NO,
        **kwargs: Any,
    ) -> None:
        """
        参数
        ----------
        time_per_step : str
            每个交易步骤的交易时间，用于生成交易日历
        show_indicator: bool, 可选
            是否显示指标，包括：
            - 'pa'：价格优势
            - 'pos'：正向率
            - 'ffr'：成交率
        indicator_config: dict, 可选
            计算交易指标的配置，包括以下字段：
            - 'show_indicator'：是否显示指标，可选，默认为False。指标包括
                - 'pa'：价格优势
                - 'pos'：正向率
                - 'ffr'：成交率
            - 'pa_config'：计算价格优势(pa)的配置，可选
                - 'base_price'：作为比较基准的价格，可选，默认为'twap'
                    - 若为'twap'，基准价格为时间加权平均价格
                    - 若为'vwap'，基准价格为成交量加权平均价格
                - 'weight_method'：计算每步不同订单pa的总交易pa时的加权方法，可选，默认为'mean'
                    - 若为'mean'，计算不同订单pa的平均值
                    - 若为'amount_weighted'，计算不同订单pa的金额加权平均值
                    - 若为'value_weighted'，计算不同订单pa的价值加权平均值
            - 'ffr_config'：计算成交率(ffr)的配置，可选
                - 'weight_method'：计算每步不同订单ffr的总交易ffr时的加权方法，可选，默认为'mean'
                    - 若为'mean'，计算不同订单ffr的平均值
                    - 若为'amount_weighted'，计算不同订单ffr的金额加权平均值
                    - 若为'value_weighted'，计算不同订单ffr的价值加权平均值
            示例：
                {
                    'show_indicator': True,
                    'pa_config': {
                        "agg": "twap",  # "vwap"
                        "price": "$close", # 默认使用交易所的成交价格
                    },
                    'ffr_config':{
                        'weight_method': 'value_weighted',
                    }
                }
        generate_portfolio_metrics : bool, 可选
            是否生成投资组合指标，默认为False
        verbose : bool, 可选
            是否打印交易信息，默认为False
        track_data : bool, 可选
            是否生成交易决策数据，用于强化学习代理训练
            - 若`self.track_data`为True，在生成训练数据时，`execute`的输入`trade_decision`将由`collect_data`生成
            - 否则，不生成`trade_decision`

        trade_exchange : Exchange
            提供市场信息的交易所，用于生成投资组合指标
            - 若generate_portfolio_metrics为None，将忽略trade_exchange
            - 否则，若`trade_exchange`为None，self.trade_exchange将通过common_infra设置

        common_infra : CommonInfrastructure, 可选:
            回测的公共基础设施，可能包括：
            - trade_account : Account, 可选
                用于交易的账户
            - trade_exchange : Exchange, 可选
                提供市场信息的交易所

        settle_type : str
            请参考BasePosition.settle_start的文档
        """
        self.time_per_step = time_per_step
        self.indicator_config = indicator_config
        self.generate_portfolio_metrics = generate_portfolio_metrics
        self.verbose = verbose
        self.track_data = track_data
        self._trade_exchange = trade_exchange
        self.level_infra = LevelInfrastructure()
        self.level_infra.reset_infra(common_infra=common_infra, executor=self)
        self._settle_type = settle_type
        self.reset(start_time=start_time, end_time=end_time, common_infra=common_infra)
        if common_infra is None:
            get_module_logger("BaseExecutor").warning(f"`common_infra` is not set for {self}")

        # 记录一天内的成交订单金额
        self.dealt_order_amount: Dict[str, float] = defaultdict(float)
        self.deal_day = None

    def reset_common_infra(self, common_infra: CommonInfrastructure, copy_trade_account: bool = False) -> None:
        """
        重置交易基础设施
            - 重置交易账户
        """
        if not hasattr(self, "common_infra"):
            self.common_infra = common_infra
        else:
            self.common_infra.update(common_infra)

        self.level_infra.reset_infra(common_infra=self.common_infra)

        if common_infra.has("trade_account"):
            # 注意: 代码中有个技巧
            # 使用浅拷贝而不是深拷贝
            # 1. 这样仓位是共享的
            # 2. 其他不共享，所以每个层级有自己的指标(组合和交易指标)
            self.trade_account: Account = (
                copy.copy(common_infra.get("trade_account"))
                if copy_trade_account
                else common_infra.get("trade_account")
            )
            self.trade_account.reset(freq=self.time_per_step, port_metr_enabled=self.generate_portfolio_metrics)

    @property
    def trade_exchange(self) -> Exchange:
        """按优先级顺序获取交易交易所"""
        return getattr(self, "_trade_exchange", None) or self.common_infra.get("trade_exchange")

    @property
    def trade_calendar(self) -> TradeCalendarManager:
        """
        尽管交易日历可以从多个来源访问，但集中管理将使代码更简洁
        """
        return self.level_infra.get("trade_calendar")

    def reset(self, common_infra: CommonInfrastructure | None = None, **kwargs: Any) -> None:
        """
        - 重置`start_time`和`end_time`，用于交易日历
        - 重置`common_infra`，用于重置`trade_account`、`trade_exchange`等
        """

        if "start_time" in kwargs or "end_time" in kwargs:
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            self.level_infra.reset_cal(freq=self.time_per_step, start_time=start_time, end_time=end_time)
        if common_infra is not None:
            self.reset_common_infra(common_infra)

    def get_level_infra(self) -> LevelInfrastructure:
        return self.level_infra

    def finished(self) -> bool:
        return self.trade_calendar.finished()

    def execute(self, trade_decision: BaseTradeDecision, level: int = 0) -> List[object]:
        """执行交易决策并返回执行结果

        注意：此函数从未在框架中直接使用。是否应该删除？

        参数
        ----------
        trade_decision : BaseTradeDecision
            交易决策

        level : int
            当前执行器的级别

        返回
        ----------
        execute_result : List[object]
            交易决策的执行结果
        """
        return_value: dict = {}
        for _decision in self.collect_data(trade_decision, return_value=return_value, level=level):
            pass
        return cast(list, return_value.get("execute_result"))

    @abstractmethod
    def _collect_data(
        self,
        trade_decision: BaseTradeDecision,
        level: int = 0,
    ) -> Union[Generator[Any, Any, Tuple[List[object], dict]], Tuple[List[object], dict]]:
        """
        请参考collect_data的文档
        `_collect_data`与`collect_data`的唯一区别是一些通用步骤已移至collect_data中

        参数
        ----------
        请参考collect_data的文档


        返回
        -------
        Tuple[List[object], dict]:
            (<交易决策的执行结果>, <`self.trade_account.update_bar_end`的额外关键字参数>)
        """

    def collect_data(
        self,
        trade_decision: BaseTradeDecision,
        return_value: dict | None = None,
        level: int = 0,
    ) -> Generator[Any, Any, List[object]]:
        """用于强化学习训练的交易决策数据收集生成器

        此函数将向前推进一个步骤

        参数
        ----------
        trade_decision : BaseTradeDecision
            交易决策

        level : int
            当前执行器的级别。0表示顶级

        return_value : dict
            用于返回值的内存地址
            例如：{"return_value": <执行结果>}

        返回
        ----------
        execute_result : List[object]
            交易决策的执行结果。
            ** 注意！！！ **:
            1) 这是必要的，生成器的返回值将在NestedExecutor中使用
            2) 请注意执行结果未合并。

        生成
        -------
        object
            交易决策
        """

        if self.track_data:
            yield trade_decision

        atomic = not issubclass(self.__class__, NestedExecutor)  # issubclass(A, A) is True

        if atomic and trade_decision.get_range_limit(default_value=None) is not None:
            raise ValueError("atomic executor doesn't support specify `range_limit`")

        if self._settle_type != BasePosition.ST_NO:
            self.trade_account.current_position.settle_start(self._settle_type)

        obj = self._collect_data(trade_decision=trade_decision, level=level)

        if isinstance(obj, GeneratorType):
            yield_res = yield from obj
            assert isinstance(yield_res, tuple) and len(yield_res) == 2
            res, kwargs = yield_res
        else:
            # Some concrete executor don't have inner decisions
            res, kwargs = obj

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time()
        # Account will not be changed in this function
        self.trade_account.update_bar_end(
            trade_start_time,
            trade_end_time,
            self.trade_exchange,
            atomic=atomic,
            outer_trade_decision=trade_decision,
            indicator_config=self.indicator_config,
            **kwargs,
        )

        self.trade_calendar.step()

        if self._settle_type != BasePosition.ST_NO:
            self.trade_account.current_position.settle_commit()

        if return_value is not None:
            return_value.update({"execute_result": res})

        return res

    def get_all_executors(self) -> List[BaseExecutor]:
        """获取所有执行器"""
        return [self]


class NestedExecutor(BaseExecutor):
    """
    具有内部策略和执行器的嵌套执行器
    - 每次调用`execute`时，它将调用内部策略和执行器在更高频率的环境中执行`trade_decision`
    """

    def __init__(
        self,
        time_per_step: str,
        inner_executor: Union[BaseExecutor, dict],
        inner_strategy: Union[BaseStrategy, dict],
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        skip_empty_decision: bool = True,
        align_range_limit: bool = True,
        common_infra: CommonInfrastructure | None = None,
        **kwargs: Any,
    ) -> None:
        """
        参数
        ----------
        inner_executor : BaseExecutor
            每个交易周期内的交易环境。
        inner_strategy : BaseStrategy
            每个交易周期内的交易策略
        skip_empty_decision: bool
            当决策为空时，执行器是否跳过调用内部循环。
            在以下情况下应设为False
            - 决策可能会逐步更新
            - 内部执行器可能不遵循外部策略的决策
        align_range_limit: bool
            强制对齐交易范围决策
            仅适用于嵌套执行器，因为range_limit由外部策略提供
        """
        self.inner_executor: BaseExecutor = init_instance_by_config(
            inner_executor,
            common_infra=common_infra,
            accept_types=BaseExecutor,
        )
        self.inner_strategy: BaseStrategy = init_instance_by_config(
            inner_strategy,
            common_infra=common_infra,
            accept_types=BaseStrategy,
        )

        self._skip_empty_decision = skip_empty_decision
        self._align_range_limit = align_range_limit

        super(NestedExecutor, self).__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_portfolio_metrics=generate_portfolio_metrics,
            verbose=verbose,
            track_data=track_data,
            common_infra=common_infra,
            **kwargs,
        )

    def reset_common_infra(self, common_infra: CommonInfrastructure, copy_trade_account: bool = False) -> None:
        """
        重置交易基础设施
            - 重置内部策略和内部执行器的公共基础设施
        """
        # 注意：请参考 BaseExecutor.reset_common_infra 的文档了解 `copy_trade_account` 的含义

        # 第一层遵循上层的 `copy_trade_account` 设置
        super(NestedExecutor, self).reset_common_infra(common_infra, copy_trade_account=copy_trade_account)

        # 下层必须复制 trade_account
        self.inner_executor.reset_common_infra(common_infra, copy_trade_account=True)
        self.inner_strategy.reset_common_infra(common_infra)

    def _init_sub_trading(self, trade_decision: BaseTradeDecision) -> None:
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time()
        self.inner_executor.reset(start_time=trade_start_time, end_time=trade_end_time)
        sub_level_infra = self.inner_executor.get_level_infra()
        self.level_infra.set_sub_level_infra(sub_level_infra)
        self.inner_strategy.reset(level_infra=sub_level_infra, outer_trade_decision=trade_decision)

    def _update_trade_decision(self, trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        # 外部策略在每次迭代时都有机会更新决策
        updated_trade_decision = trade_decision.update(self.inner_executor.trade_calendar)
        if updated_trade_decision is not None:  # TODO: 目前总是为 None？
            trade_decision = updated_trade_decision
            # 新更新
            # 为内部策略创建一个钩子来更新外部决策
            trade_decision = self.inner_strategy.alter_outer_trade_decision(trade_decision)
        return trade_decision

    def _collect_data(
        self,
        trade_decision: BaseTradeDecision,
        level: int = 0,
    ) -> Generator[Any, Any, Tuple[List[object], dict]]:
        execute_result = []
        inner_order_indicators = []
        decision_list = []
        # 注意：
        # - 这对于计算子层级的步骤是必要的
        # - 更详细的信息将被设置到交易决策中
        self._init_sub_trading(trade_decision)

        _inner_execute_result = None
        while not self.inner_executor.finished():
            trade_decision = self._update_trade_decision(trade_decision)

            if trade_decision.empty() and self._skip_empty_decision:
                # give one chance for outer strategy to update the strategy
                # - For updating some information in the sub executor (the strategy have no knowledge of the inner
                #   executor when generating the decision)
                break

            sub_cal: TradeCalendarManager = self.inner_executor.trade_calendar

            # 注意：确保get_start_end_idx在`self._update_trade_decision`之后调用
            start_idx, end_idx = get_start_end_idx(sub_cal, trade_decision)
            if not self._align_range_limit or start_idx <= sub_cal.get_trade_step() <= end_idx:
                # if force align the range limit, skip the steps outside the decision range limit

                res = self.inner_strategy.generate_trade_decision(_inner_execute_result)

                # 注意：!!!!!
                # 下面两行代码是针对RL的特殊情况
                # 用于解决以下冲突：
                # - 通常用户会创建一个策略并嵌入到Qlib的执行器和模拟器交互循环中
                #   例如一个嵌套Qlib示例：(Qlib策略) <=> (Qlib执行器[(内部Qlib策略) <=> (内部Qlib执行器)])
                # - 然而基于RL的框架有自己的脚本来运行循环
                #   例如一个RL学习示例：(RL策略) <=> (RL环境[(内部Qlib执行器)])
                # 为了使嵌套Qlib示例和RL学习示例能够一起运行，提出以下解决方案：
                # - 入口脚本遵循RL学习示例的格式，以兼容各种RL框架
                # - RL环境的每一步都会让内部Qlib执行器前进一步
                #     - 内部Qlib策略是一个代理策略，它会通过`yield from`将程序控制权交给RL环境
                #       并等待策略的动作
                # 所以下面两行代码是实现控制权转移
                if isinstance(res, GeneratorType):
                    res = yield from res

                _inner_trade_decision: BaseTradeDecision = res

                trade_decision.mod_inner_decision(_inner_trade_decision)  # propagate part of decision information

                # 注意：必须在collect_data之前调用sub_cal.get_step_time()，以防步骤偏移
                decision_list.append((_inner_trade_decision, *sub_cal.get_step_time()))

                # 注意：交易日历将在下一行前进
                _inner_execute_result = yield from self.inner_executor.collect_data(
                    trade_decision=_inner_trade_decision,
                    level=level + 1,
                )
                assert isinstance(_inner_execute_result, list)
                self.post_inner_exe_step(_inner_execute_result)
                execute_result.extend(_inner_execute_result)

                inner_order_indicators.append(
                    self.inner_executor.trade_account.get_trade_indicator().get_order_indicator(raw=True),
                )
            else:
                # do nothing and just step forward
                sub_cal.step()

        # Let inner strategy know that the outer level execution is done.
        self.inner_strategy.post_upper_level_exe_step()

        return execute_result, {"inner_order_indicators": inner_order_indicators, "decision_list": decision_list}

    def post_inner_exe_step(self, inner_exe_res: List[object]) -> None:
        """
        A hook for doing sth after each step of inner strategy

        Parameters
        ----------
        inner_exe_res :
            the execution result of inner task
        """
        self.inner_strategy.post_exe_step(inner_exe_res)

    def get_all_executors(self) -> List[BaseExecutor]:
        """get all executors, including self and inner_executor.get_all_executors()"""
        return [self, *self.inner_executor.get_all_executors()]


def _retrieve_orders_from_decision(trade_decision: BaseTradeDecision) -> List[Order]:
    """
    IDE-friendly helper function.
    """
    decisions = trade_decision.get_decision()
    orders: List[Order] = []
    for decision in decisions:
        assert isinstance(decision, Order)
        orders.append(decision)
    return orders


class SimulatorExecutor(BaseExecutor):
    """Executor that simulate the true market"""

    # TODO: TT_SERIAL & TT_PARAL will be replaced by feature fix_pos now.
    # Please remove them in the future.

    # available trade_types
    TT_SERIAL = "serial"
    # The orders will be executed serially in a sequence
    # In each trading step, it is possible that users sell instruments first and use the money to buy new instruments
    TT_PARAL = "parallel"
    # The orders will be executed in parallel
    # In each trading step, if users try to sell instruments first and buy new instruments with money, failure will
    # occur

    def __init__(
        self,
        time_per_step: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        common_infra: CommonInfrastructure | None = None,
        trade_type: str = TT_SERIAL,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        trade_type: str
            please refer to the doc of `TT_SERIAL` & `TT_PARAL`
        """
        super(SimulatorExecutor, self).__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_portfolio_metrics=generate_portfolio_metrics,
            verbose=verbose,
            track_data=track_data,
            common_infra=common_infra,
            **kwargs,
        )

        self.trade_type = trade_type

    def _get_order_iterator(self, trade_decision: BaseTradeDecision) -> List[Order]:
        """

        参数
        ----------
        trade_decision : BaseTradeDecision
            策略给出的交易决策

        返回
        -------
        List[Order]:
            根据`self.trade_type`获取订单列表
        """
        orders = _retrieve_orders_from_decision(trade_decision)

        if self.trade_type == self.TT_SERIAL:
            # 订单将以并行方式进行交易
            order_it = orders
        elif self.trade_type == self.TT_PARAL:
            # NOTE: !!!!!!!
            # Assumption: there will not be orders in different trading direction in a single step of a strategy !!!!
            # The parallel trading failure will be caused only by the conflicts of money
            # Therefore, make the buying go first will make sure the conflicts happen.
            # It equals to parallel trading after sorting the order by direction
            order_it = sorted(orders, key=lambda order: -order.direction)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return order_it

    def _collect_data(self, trade_decision: BaseTradeDecision, level: int = 0) -> Tuple[List[object], dict]:
        """
        执行交易决策并返回执行结果和交易信息

        参数
        ----------
        trade_decision : BaseTradeDecision
            策略生成的交易决策
        level : int
            当前执行器的级别

        返回
        -------
        Tuple[List[object], dict]:
            包含执行结果和交易信息的元组
        """
        trade_start_time, _ = self.trade_calendar.get_step_time()
        execute_result: list = []

        for order in self._get_order_iterator(trade_decision):
            # Each time we move into a new date, clear `self.dealt_order_amount` since it only maintains intraday
            # information.
            now_deal_day = self.trade_calendar.get_step_time()[0].floor(freq="D")
            if self.deal_day is None or now_deal_day > self.deal_day:
                self.dealt_order_amount = defaultdict(float)
                self.deal_day = now_deal_day

            # execute the order.
            # NOTE: The trade_account will be changed in this function
            trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                order,
                trade_account=self.trade_account,
                dealt_order_amount=self.dealt_order_amount,
            )
            execute_result.append((order, trade_val, trade_cost, trade_price))

            self.dealt_order_amount[order.stock_id] += order.deal_amount

            if self.verbose:
                print(
                    "[I {:%Y-%m-%d %H:%M:%S}]: {} {}, price {:.2f}, amount {}, deal_amount {}, factor {}, "
                    "value {:.2f}, cash {:.2f}.".format(
                        trade_start_time,
                        "sell" if order.direction == Order.SELL else "buy",
                        order.stock_id,
                        trade_price,
                        order.amount,
                        order.deal_amount,
                        order.factor,
                        trade_val,
                        self.trade_account.get_cash(),
                    ),
                )
        return execute_result, {"trade_info": execute_result}
