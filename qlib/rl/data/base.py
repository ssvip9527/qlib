# 版权所有 (c) 微软公司。
# MIT许可证授权。
from __future__ import annotations

from abc import abstractmethod

import pandas as pd


class BaseIntradayBacktestData:
    """
    常用于回测的原始市场数据（因此称为回测数据）。

    所有类型回测数据的基类。目前，每种类型的模拟器都有其对应的回测数据类型。
    """

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_deal_price(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_volume(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_time_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError


class BaseIntradayProcessedData:
    """经过数据清洗和特征工程处理后的市场数据。

    它包含“今日”和“昨日”的处理数据，因为某些算法可能会使用前一天的市场信息来辅助决策。
    """

    today: pd.DataFrame
    """“今日”的处理数据。
    记录数必须为 ``time_length``，列数必须为 ``feature_dim``。"""

    yesterday: pd.DataFrame
    """“昨日”的处理数据。
    记录数必须为 ``time_length``，列数必须为 ``feature_dim``。"""


class ProcessedDataProvider:
    """处理后数据的提供器"""

    def get_data(
        self,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> BaseIntradayProcessedData:
        raise NotImplementedError
