# Copyright (c) Microsoft Corporation.
# MIT许可证授权。

from __future__ import annotations

import typing
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from qlib.backtest import Order
from qlib.typehint import TypedDict

if typing.TYPE_CHECKING:
    from qlib.rl.data.base import BaseIntradayBacktestData


class SAOEMetrics(TypedDict):
    """SAOE(单资产订单执行)的指标数据，可累计计算一个"周期"内的指标。
    可以按天累计，或按时间段(如30分钟)累计，或每分钟单独计算。

    警告
    --------
    类型提示是针对单个元素的，但很多时候这些指标可以是向量化的。
    例如，``market_volume``可以是一个浮点数列表(或ndarray)而不仅是单个浮点数。
    """

    stock_id: str
    """该记录的股票ID。"""
    datetime: pd.Timestamp | pd.DatetimeIndex
    """该记录的时间戳(在数据框中作为索引)。"""
    direction: int
    """订单方向。0表示卖出，1表示买入。"""

    # Market information.
    market_volume: np.ndarray | float
    """该时间段内的(总)市场成交量。"""
    market_price: np.ndarray | float
    """成交价格。如果是时间段，则为该时间段内的平均市场成交价。"""

    # Strategy records.

    amount: np.ndarray | float
    """策略计划交易的总量(成交量)。"""
    inner_amount: np.ndarray | float
    """下层策略计划交易的总量
    (可能大于amount，例如为了确保ffr)。"""

    deal_amount: np.ndarray | float
    """实际生效的交易量(必须小于inner_amount)。"""
    trade_price: np.ndarray | float
    """该策略的平均成交价格。"""
    trade_value: np.ndarray | float
    """交易总价值。在简单模拟中，trade_value = deal_amount * price。"""
    position: np.ndarray | float
    """该"周期"后剩余的持仓量。"""

    # Accumulated metrics

    ffr: np.ndarray | float
    """已完成每日订单的百分比。"""

    pa: np.ndarray | float
    """与基准(即使用基准市场价格交易)相比的价格优势。
    基准是使用TWAP策略执行该订单时的交易价格。
    请注意这里可能存在数据泄漏。
    单位为BP(基点，1/10000)。"""


class SAOEState(NamedTuple):
    """SAOE(单资产订单执行)模拟器的状态数据结构。"""

    order: Order
    """正在处理的订单。"""
    cur_time: pd.Timestamp
    """当前时间，例如9:30。"""
    cur_step: int
    """当前步骤，例如0。"""
    position: float
    """当前剩余待执行的交易量。"""
    history_exec: pd.DataFrame
    """参见 :attr:`SingleAssetOrderExecution.history_exec`。"""
    history_steps: pd.DataFrame
    """参见 :attr:`SingleAssetOrderExecution.history_steps`。"""

    metrics: Optional[SAOEMetrics]
    """每日指标，仅在交易处于"完成"状态时可用。"""

    backtest_data: BaseIntradayBacktestData
    """状态中包含回测数据。
    实际上，目前只需要该数据的时间索引。
    包含完整数据是为了支持依赖原始数据的算法(如VWAP)的实现。
    解释器可以按需使用这些数据，但应注意避免泄漏未来数据。
    """

    ticks_per_step: int
    """每个步骤包含多少个tick。"""
    ticks_index: pd.DatetimeIndex
    """全天的交易tick，未按订单切片(在数据中定义)。例如[9:30, 9:31, ..., 14:59]。"""
    ticks_for_order: pd.DatetimeIndex
    """按订单切片的交易tick，例如[9:45, 9:46, ..., 14:44]。"""
