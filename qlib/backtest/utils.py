# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Set, Tuple, TYPE_CHECKING, Union

import numpy as np

from qlib.utils.time import epsilon_change

if TYPE_CHECKING:
    from qlib.backtest.decision import BaseTradeDecision

import warnings

import pandas as pd

from ..data.data import Cal


class TradeCalendarManager:
    """
    交易日历管理器
        - BaseStrategy和BaseExecutor会使用该类
    """

    def __init__(
        self,
        freq: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        level_infra: LevelInfrastructure | None = None,
    ) -> None:
        """
        参数
        ----------
        freq : str
            交易日历频率，也是每个交易步骤的交易时间
        start_time : Union[str, pd.Timestamp], optional
            交易日历的闭区间起始时间，默认为None
            如果`start_time`为None，必须在交易前重置
        end_time : Union[str, pd.Timestamp], optional
            交易时间范围的闭区间结束时间，默认为None
            如果`end_time`为None，必须在交易前重置
        """
        self.level_infra = level_infra
        self.reset(freq=freq, start_time=start_time, end_time=end_time)

    def reset(
        self,
        freq: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
    ) -> None:
        """
        请参考`__init__`方法的文档

        重置交易日历
        - self.trade_len : 交易步骤总数
        - self.trade_step : 已完成的交易步骤数，self.trade_step取值范围为
            [0, 1, 2, ..., self.trade_len - 1]
        """
        self.freq = freq
        self.start_time = pd.Timestamp(start_time) if start_time else None
        self.end_time = pd.Timestamp(end_time) if end_time else None

        _calendar = Cal.calendar(freq=freq, future=True)
        assert isinstance(_calendar, np.ndarray)
        self._calendar = _calendar
        _, _, _start_index, _end_index = Cal.locate_index(start_time, end_time, freq=freq, future=True)
        self.start_index = _start_index
        self.end_index = _end_index
        self.trade_len = _end_index - _start_index + 1
        self.trade_step = 0

    def finished(self) -> bool:
        """
        检查交易是否结束
        - 在调用strategy.generate_decisions和executor.execute前应检查
        - 如果self.trade_step >= self.self.trade_len，表示交易已完成
        - 如果self.trade_step < self.self.trade_len，表示已完成的交易步骤数为self.trade_step
        """
        return self.trade_step >= self.trade_len

    def step(self) -> None:
        if self.finished():
            raise RuntimeError(f"The calendar is finished, please reset it if you want to call it!")
        self.trade_step += 1

    def get_freq(self) -> str:
        return self.freq

    def get_trade_len(self) -> int:
        """获取总步骤长度"""
        return self.trade_len

    def get_trade_step(self) -> int:
        return self.trade_step

    def get_step_time(self, trade_step: int | None = None, shift: int = 0) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        获取第trade_step个交易区间的左右端点

        关于端点:
            - Qlib在时间序列数据选择中使用闭区间，与pandas.Series.loc性能相同
            # - 返回的右端点应减去1秒，因为Qlib中使用闭区间表示
            # 注意：Qlib支持分钟级决策执行，所以1秒小于任何交易时间间隔

        参数
        ----------
        trade_step : int, optional
            已完成的交易步骤数，默认为None表示当前步骤
        shift : int, optional
            偏移的bar数，默认为0

        返回值
        -------
        Tuple[pd.Timestamp, pd.Timestamp]
            - 如果shift == 0，返回当前交易时间范围
            - 如果shift > 0，返回前shift个bar的交易时间范围
            - 如果shift < 0，返回后shift个bar的交易时间范围
        """
        if trade_step is None:
            trade_step = self.get_trade_step()
        calendar_index = self.start_index + trade_step - shift
        return self._calendar[calendar_index], epsilon_change(self._calendar[calendar_index + 1])

    def get_data_cal_range(self, rtype: str = "full") -> Tuple[int, int]:
        """
        获取日历范围
        做出以下假设：
        1) common_infra中的交易所频率与数据日历相同
        2) 用户希望按天(即240分钟)对**数据索引**取模

        参数
        ----------
        rtype: str
            - "full": 返回当天决策的完整限制范围
            - "step": 返回当前步骤的限制范围

        返回值
        -------
        Tuple[int, int]:
        """
        # potential performance issue
        assert self.level_infra is not None

        day_start = pd.Timestamp(self.start_time.date())
        day_end = epsilon_change(day_start + pd.Timedelta(days=1))
        freq = self.level_infra.get("common_infra").get("trade_exchange").freq
        _, _, day_start_idx, _ = Cal.locate_index(day_start, day_end, freq=freq)

        if rtype == "full":
            _, _, start_idx, end_index = Cal.locate_index(self.start_time, self.end_time, freq=freq)
        elif rtype == "step":
            _, _, start_idx, end_index = Cal.locate_index(*self.get_step_time(), freq=freq)
        else:
            raise ValueError(f"This type of input {rtype} is not supported")

        return start_idx - day_start_idx, end_index - day_start_idx

    def get_all_time(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """获取交易的开始时间和结束时间"""
        return self.start_time, self.end_time

    # helper functions
    def get_range_idx(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[int, int]:
        """
        获取包含start_time~end_time范围的索引(两边都是闭区间)

        参数
        ----------
        start_time : pd.Timestamp
            开始时间
        end_time : pd.Timestamp
            结束时间

        返回值
        -------
        Tuple[int, int]:
            范围的索引。**左右都是闭区间**
        """
        left = int(np.searchsorted(self._calendar, start_time, side="right") - 1)
        right = int(np.searchsorted(self._calendar, end_time, side="right") - 1)
        left -= self.start_index
        right -= self.start_index

        def clip(idx: int) -> int:
            return min(max(0, idx), self.trade_len - 1)

        return clip(left), clip(right)

    def __repr__(self) -> str:
        return (
            f"class: {self.__class__.__name__}; "
            f"{self.start_time}[{self.start_index}]~{self.end_time}[{self.end_index}]: "
            f"[{self.trade_step}/{self.trade_len}]"
        )


class BaseInfrastructure:
    def __init__(self, **kwargs: Any) -> None:
        self.reset_infra(**kwargs)

    @abstractmethod
    def get_support_infra(self) -> Set[str]:
        raise NotImplementedError("`get_support_infra` is not implemented!")

    def reset_infra(self, **kwargs: Any) -> None:
        support_infra = self.get_support_infra()
        for k, v in kwargs.items():
            if k in support_infra:
                setattr(self, k, v)
            else:
                warnings.warn(f"{k} is ignored in `reset_infra`!")

    def get(self, infra_name: str) -> Any:
        if hasattr(self, infra_name):
            return getattr(self, infra_name)
        else:
            warnings.warn(f"infra {infra_name} is not found!")

    def has(self, infra_name: str) -> bool:
        return infra_name in self.get_support_infra() and hasattr(self, infra_name)

    def update(self, other: BaseInfrastructure) -> None:
        support_infra = other.get_support_infra()
        infra_dict = {_infra: getattr(other, _infra) for _infra in support_infra if hasattr(other, _infra)}
        self.reset_infra(**infra_dict)


class CommonInfrastructure(BaseInfrastructure):
    def get_support_infra(self) -> Set[str]:
        return {"trade_account", "trade_exchange"}


class LevelInfrastructure(BaseInfrastructure):
    """层级基础设施由执行器创建，然后共享给同层级的策略"""

    def get_support_infra(self) -> Set[str]:
        """
        关于基础设施的描述

        sub_level_infra:
        - **注意**: 仅在_init_sub_trading之后才会生效!!!
        """
        return {"trade_calendar", "sub_level_infra", "common_infra", "executor"}

    def reset_cal(
        self,
        freq: str,
        start_time: Union[str, pd.Timestamp, None],
        end_time: Union[str, pd.Timestamp, None],
    ) -> None:
        """重置交易日历管理器"""
        if self.has("trade_calendar"):
            self.get("trade_calendar").reset(freq, start_time=start_time, end_time=end_time)
        else:
            self.reset_infra(
                trade_calendar=TradeCalendarManager(freq, start_time=start_time, end_time=end_time, level_infra=self),
            )

    def set_sub_level_infra(self, sub_level_infra: LevelInfrastructure) -> None:
        """这将使跨多层级访问日历更加方便"""
        self.reset_infra(sub_level_infra=sub_level_infra)


def get_start_end_idx(trade_calendar: TradeCalendarManager, outer_trade_decision: BaseTradeDecision) -> Tuple[int, int]:
    """
    用于获取内部策略决策级别索引范围限制的辅助函数
    - 注意: 此函数不适用于订单级别

    参数
    ----------
    trade_calendar : TradeCalendarManager
        交易日历管理器
    outer_trade_decision : BaseTradeDecision
        外部策略做出的交易决策

    返回值
    -------
    Union[int, int]:
        开始索引和结束索引
    """
    try:
        return outer_trade_decision.get_range_limit(inner_calendar=trade_calendar)
    except NotImplementedError:
        return 0, trade_calendar.get_trade_len() - 1
