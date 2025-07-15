# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 区域常量
from typing import TypeVar

import numpy as np
import pandas as pd

REG_CN = "cn"
REG_US = "us"
REG_TW = "tw"

# 用于避免除零错误的极小值。
EPS = 1e-12

# 整数类型的无穷大
INF = int(1e18)
ONE_DAY = pd.Timedelta("1day")
ONE_MIN = pd.Timedelta("1min")
EPS_T = pd.Timedelta("1s")  # 使用1秒来排除区间右端点
float_or_ndarray = TypeVar("float_or_ndarray", float, np.ndarray)
