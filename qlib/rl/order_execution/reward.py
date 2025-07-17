# 版权所有 (c) 微软公司。
# MIT许可证授权。

from __future__ import annotations

from typing import cast

import numpy as np

from qlib.backtest.decision import OrderDir
from qlib.rl.order_execution.state import SAOEMetrics, SAOEState
from qlib.rl.reward import Reward

__all__ = ["PAPenaltyReward"]


class PAPenaltyReward(Reward[SAOEState]):
    """鼓励更高的PA(价格优势)，但对在短时间内堆积所有交易量进行惩罚。
    形式上，每个时间步的奖励是 :math:`(PA_t * vol_t / target - vol_t^2 * penalty)`。

    参数
    ----------
    penalty
        短时间内大交易量的惩罚系数。
    scale
        用于放大或缩小奖励的权重。
    """

    def __init__(self, penalty: float = 100.0, scale: float = 1.0) -> None:
        self.penalty = penalty
        self.scale = scale

    def reward(self, simulator_state: SAOEState) -> float:
        whole_order = simulator_state.order.amount
        assert whole_order > 0
        last_step = cast(SAOEMetrics, simulator_state.history_steps.reset_index().iloc[-1].to_dict())
        pa = last_step["pa"] * last_step["amount"] / whole_order

        # Inspect the "break-down" of the latest step: trading amount at every tick
        last_step_breakdown = simulator_state.history_exec.loc[last_step["datetime"] :]
        penalty = -self.penalty * ((last_step_breakdown["amount"] / whole_order) ** 2).sum()

        reward = pa + penalty

        # Throw error in case of NaN
        assert not (np.isnan(reward) or np.isinf(reward)), f"Invalid reward for simulator state: {simulator_state}"

        self.log("reward/pa", pa)
        self.log("reward/penalty", penalty)
        return reward * self.scale


class PPOReward(Reward[SAOEState]):
    """基于论文《基于近端策略优化的端到端最优交易执行框架》提出的奖励函数。

    参数
    ----------
    max_step
        最大步数。
    start_time_index
        允许交易的最早时间索引。
    end_time_index
        允许交易的最晚时间索引。
    """

    def __init__(self, max_step: int, start_time_index: int = 0, end_time_index: int = 239) -> None:
        self.max_step = max_step
        self.start_time_index = start_time_index
        self.end_time_index = end_time_index

    def reward(self, simulator_state: SAOEState) -> float:
        if simulator_state.cur_step == self.max_step - 1 or simulator_state.position < 1e-6:
            if simulator_state.history_exec["deal_amount"].sum() == 0.0:
                vwap_price = cast(
                    float,
                    np.average(simulator_state.history_exec["market_price"]),
                )
            else:
                vwap_price = cast(
                    float,
                    np.average(
                        simulator_state.history_exec["market_price"],
                        weights=simulator_state.history_exec["deal_amount"],
                    ),
                )
            twap_price = simulator_state.backtest_data.get_deal_price().mean()

            if simulator_state.order.direction == OrderDir.SELL:
                ratio = vwap_price / twap_price if twap_price != 0 else 1.0
            else:
                ratio = twap_price / vwap_price if vwap_price != 0 else 1.0
            if ratio < 1.0:
                return -1.0
            elif ratio < 1.1:
                return 0.0
            else:
                return 1.0
        else:
            return 0.0
