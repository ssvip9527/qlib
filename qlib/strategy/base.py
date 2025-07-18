# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Generator, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.position import BasePosition
    from qlib.backtest.executor import BaseExecutor

from typing import Tuple

from ..backtest.decision import BaseTradeDecision
from ..backtest.utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager
from ..rl.interpreter import ActionInterpreter, StateInterpreter
from ..utils import init_instance_by_config

__all__ = ["BaseStrategy", "RLStrategy", "RLIntStrategy"]


class BaseStrategy:
    """交易策略基类"""

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        trade_exchange: Exchange = None,
    ) -> None:
        """
        参数
        ----------
        outer_trade_decision : BaseTradeDecision, optional
            本策略依赖的外部策略交易决策，将在[start_time, end_time]区间内交易，默认为None

            - 如果策略用于拆分交易决策，将会使用此参数
            - 如果策略用于投资组合管理，可以忽略此参数
        level_infra : LevelInfrastructure, optional
            回测共享的层级基础设施，包括交易日历等
        common_infra : CommonInfrastructure, optional
            回测共享的公共基础设施，包括交易账户、交易交易所等

        trade_exchange : Exchange
            提供市场信息的交易所，用于处理订单和生成报告

            - 如果`trade_exchange`为None，self.trade_exchange将从common_infra中获取
            - 允许在不同的执行中使用不同的交易所
            - 例如：

                - 在日线执行中，日线交易所和分钟线交易所都可用，但推荐使用日线交易所，因为它运行更快
                - 在分钟线执行中，日线交易所不可用，只能使用分钟线交易所
        """

        self._reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision)
        self._trade_exchange = trade_exchange

    @property
    def executor(self) -> BaseExecutor:
        return self.level_infra.get("executor")

    @property
    def trade_calendar(self) -> TradeCalendarManager:
        return self.level_infra.get("trade_calendar")

    @property
    def trade_position(self) -> BasePosition:
        return self.common_infra.get("trade_account").current_position

    @property
    def trade_exchange(self) -> Exchange:
        """按优先级顺序获取交易交易所"""
        return getattr(self, "_trade_exchange", None) or self.common_infra.get("trade_exchange")

    def reset_level_infra(self, level_infra: LevelInfrastructure) -> None:
        if not hasattr(self, "level_infra"):
            self.level_infra = level_infra
        else:
            self.level_infra.update(level_infra)

    def reset_common_infra(self, common_infra: CommonInfrastructure) -> None:
        if not hasattr(self, "common_infra"):
            self.common_infra: CommonInfrastructure = common_infra
        else:
            self.common_infra.update(common_infra)

    def reset(
        self,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        outer_trade_decision: BaseTradeDecision = None,
        **kwargs,
    ) -> None:
        """
        - 重置`level_infra`，用于重置交易日历等
        - 重置`common_infra`，用于重置`trade_account`、`trade_exchange`等
        - 重置`outer_trade_decision`，用于做出拆分决策

        **注意**:
        将此函数拆分为`reset`和`_reset`将使以下情况更方便
        1. 用户希望通过重写`reset`来初始化策略，但不想影响初始化时调用的`_reset`
        """
        self._reset(
            level_infra=level_infra,
            common_infra=common_infra,
            outer_trade_decision=outer_trade_decision,
        )

    def _reset(
        self,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        outer_trade_decision: BaseTradeDecision = None,
    ):
        """
        请参考`reset`的文档
        """
        if level_infra is not None:
            self.reset_level_infra(level_infra)

        if common_infra is not None:
            self.reset_common_infra(common_infra)

        if outer_trade_decision is not None:
            self.outer_trade_decision = outer_trade_decision

    @abstractmethod
    def generate_trade_decision(
        self,
        execute_result: list = None,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        """在每个交易bar生成交易决策

        参数
        ----------
        execute_result : List[object], optional
            交易决策的执行结果，默认为None

            - 首次调用generate_trade_decision时，`execute_result`可能为None
        """
        raise NotImplementedError("generate_trade_decision is not implemented!")

    # helper methods: not necessary but for convenience
    def get_data_cal_avail_range(self, rtype: str = "full") -> Tuple[int, int]:
        """
        返回`self`策略的数据日历可用决策范围
        该范围考虑以下因素：
        - `self`策略负责的数据日历
        - 外部策略决策的交易范围限制

        相关方法
        - TradeCalendarManager.get_data_cal_range
        - BaseTradeDecision.get_data_cal_range_limit

        参数
        ----------
        rtype: str
            - "full": 返回策略从`start_time`到`end_time`的可用数据索引范围
            - "step": 返回策略当前步骤的可用数据索引范围

        返回
        -------
        Tuple[int, int]:
            可用的范围，两端均为闭区间
        """
        cal_range = self.trade_calendar.get_data_cal_range(rtype=rtype)
        if self.outer_trade_decision is None:
            raise ValueError(f"There is not limitation for strategy {self}")
        range_limit = self.outer_trade_decision.get_data_cal_range_limit(rtype=rtype)
        return max(cal_range[0], range_limit[0]), min(cal_range[1], range_limit[1])

    """
    The following methods are used to do cross-level communications in nested execution.
    You do not need to care about them if you are implementing a single-level execution.
    """

    @staticmethod
    def update_trade_decision(
        trade_decision: BaseTradeDecision,
        trade_calendar: TradeCalendarManager,
    ) -> Optional[BaseTradeDecision]:
        """
        在内部执行的每个步骤中更新交易决策，此方法启用所有订单

        参数
        ----------
        trade_decision : BaseTradeDecision
            将被更新的交易决策
        trade_calendar : TradeCalendarManager
            **内部策略**的日历!!!!!

        返回
        -------
            BaseTradeDecision:
        """
        # default to return None, which indicates that the trade decision is not changed
        return None

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        """
        A method for updating the outer_trade_decision.
        The outer strategy may change its decision during updating.

        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision
            the decision updated by the outer strategy

        Returns
        -------
            BaseTradeDecision
        """
        # default to reset the decision directly
        # NOTE: normally, user should do something to the strategy due to the change of outer decision
        return outer_trade_decision

    def post_upper_level_exe_step(self) -> None:
        """
        A hook for doing sth after the upper level executor finished its execution (for example, finalize
        the metrics collection).
        """

    def post_exe_step(self, execute_result: Optional[list]) -> None:
        """
        A hook for doing sth after the corresponding executor finished its execution.

        Parameters
        ----------
        execute_result :
            the execution result
        """


class RLStrategy(BaseStrategy, metaclass=ABCMeta):
    """RL-based strategy"""

    def __init__(
        self,
        policy,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        policy :
            RL policy for generate action
        """
        super(RLStrategy, self).__init__(outer_trade_decision, level_infra, common_infra, **kwargs)
        self.policy = policy


class RLIntStrategy(RLStrategy, metaclass=ABCMeta):
    """(RL)-based (Strategy) with (Int)erpreter"""

    def __init__(
        self,
        policy,
        state_interpreter: dict | StateInterpreter,
        action_interpreter: dict | ActionInterpreter,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        state_interpreter : Union[dict, StateInterpreter]
            interpreter that interprets the qlib execute result into rl env state
        action_interpreter : Union[dict, ActionInterpreter]
            interpreter that interprets the rl agent action into qlib order list
        start_time : Union[str, pd.Timestamp], optional
            start time of trading, by default None
        end_time : Union[str, pd.Timestamp], optional
            end time of trading, by default None
        """
        super(RLIntStrategy, self).__init__(policy, outer_trade_decision, level_infra, common_infra, **kwargs)

        self.policy = policy
        self.state_interpreter = init_instance_by_config(state_interpreter, accept_types=StateInterpreter)
        self.action_interpreter = init_instance_by_config(action_interpreter, accept_types=ActionInterpreter)

    def generate_trade_decision(self, execute_result: list = None) -> BaseTradeDecision:
        _interpret_state = self.state_interpreter.interpret(execute_result=execute_result)
        _action = self.policy.step(_interpret_state)
        _trade_decision = self.action_interpreter.interpret(action=_action)
        return _trade_decision
