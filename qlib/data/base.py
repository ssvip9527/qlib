# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import abc
import pandas as pd
from ..log import get_module_logger


class Expression(abc.ABC):
    """表达式基类

    表达式用于处理具有以下格式的数据计算
    每个工具包含两个维度的数据：

    - 特征（feature）
    - 时间（time）：可以是观察时间或周期时间
        - 周期时间专为时点数据库设计。例如，周期时间可能是2014Q4，其值可以被多次观察（由于修正，不同时间可能观察到不同值）。
    """

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        from .ops import Gt  # pylint: disable=C0415

        return Gt(self, other)

    def __ge__(self, other):
        from .ops import Ge  # pylint: disable=C0415

        return Ge(self, other)

    def __lt__(self, other):
        from .ops import Lt  # pylint: disable=C0415

        return Lt(self, other)

    def __le__(self, other):
        from .ops import Le  # pylint: disable=C0415

        return Le(self, other)

    def __eq__(self, other):
        from .ops import Eq  # pylint: disable=C0415

        return Eq(self, other)

    def __ne__(self, other):
        from .ops import Ne  # pylint: disable=C0415

        return Ne(self, other)

    def __add__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(self, other)

    def __radd__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(other, self)

    def __sub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(self, other)

    def __rsub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(other, self)

    def __mul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __rmul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __div__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rdiv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __truediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __pow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(self, other)

    def __rpow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(other, self)

    def __and__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(self, other)

    def __rand__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(other, self)

    def __or__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(self, other)

    def __ror__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(other, self)

    def load(self, instrument, start_index, end_index, *args):
        """加载特征
        此函数负责基于表达式引擎加载特征/表达式。

        具体实现分为两部分：
            1) 缓存数据，处理错误。
                - 这部分由所有表达式共享，并在Expression中实现
            2) 根据特定表达式处理和计算数据。
                - 这部分在每个表达式中不同，并在每个表达式中实现

        表达式引擎由不同数据共享。
        不同数据会为`args`提供不同的额外信息。

        参数
        ----------
        instrument : str
            工具代码
        start_index : str
            特征开始索引[在日历中]
        end_index : str
            特征结束索引[在日历中]

        args可能包含以下信息：
            1) 如果用于基本表达式引擎数据，包含以下参数
                freq: str
                    特征频率
            2) 如果用于PIT数据，包含以下参数
                cur_pit:
                    专为时点数据设计
                period: int
                    用于查询特定周期
                    Qlib中周期用整数表示（例如202001可能表示2020年第一季度）

        返回
        ----------
        pd.Series
            特征序列：序列的索引是日历索引
        """
        from .cache import H  # pylint: disable=C0415

        # 缓存
        cache_key = str(self), instrument, start_index, end_index, *args
        if cache_key in H["f"]:
            return H["f"][cache_key]
        if start_index is not None and end_index is not None and start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        try:
            series = self._load_internal(instrument, start_index, end_index, *args)
        except Exception as e:
            get_module_logger("data").debug(
                f"加载数据错误: instrument={instrument}, expression={str(self)}, "
                f"start_index={start_index}, end_index={end_index}, args={args}. "
                f"错误信息: {str(e)}"
            )
            raise
        series.name = str(self)
        H["f"][cache_key] = series
        return series

    @abc.abstractmethod
    def _load_internal(self, instrument, start_index, end_index, *args) -> pd.Series:
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_longest_back_rolling(self):
        """获取特征需要访问的最长历史数据长度

        该方法设计用于预先获取计算特定范围内特征所需的数据范围。
        但类似Ref(Ref($close, -1), 1)的情况无法正确处理。

        因此该方法仅用于检测所需的历史数据长度。
        """
        # TODO: forward operator like Ref($close, -1) is not supported yet.
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_extended_window_size(self):
        """获取扩展窗口大小

        为了在范围[start_index, end_index]内计算该运算符，
        我们需要获取*叶子特征*在范围
        [start_index - lft_etd, end_index + rght_etd]内的值。

        返回
        ----------
        (int, int)
            左扩展长度, 右扩展长度
        """
        raise NotImplementedError("This function must be implemented in your newly defined feature")


class Feature(Expression):
    """静态表达式

    此类特征将从数据提供者加载数据
    """

    def __init__(self, name=None):
        if name:
            self._name = name
        else:
            self._name = type(self).__name__

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument, start_index, end_index, freq):
        # load
        from .data import FeatureD  # pylint: disable=C0415

        return FeatureD.feature(instrument, str(self), start_index, end_index, freq)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class PFeature(Feature):
    def __str__(self):
        return "$$" + self._name

    def _load_internal(self, instrument, start_index, end_index, cur_time, period=None):
        from .data import PITD  # pylint: disable=C0415

        return PITD.period_feature(instrument, str(self), start_index, end_index, cur_time, period)


class ExpressionOps(Expression):
    """运算符表达式

    此类特征将动态使用运算符进行特征构建。
    """
