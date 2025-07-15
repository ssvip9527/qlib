# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pathlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union, cast

import numpy as np
import pandas as pd

import qlib.utils.index_data as idd
from qlib.backtest.decision import BaseTradeDecision, Order, OrderDir
from qlib.backtest.exchange import Exchange

from ..tests.config import CSI300_BENCH
from ..utils.resam import get_higher_eq_freq_feature, resam_ts_data
from .high_performance_ds import BaseOrderIndicator, BaseSingleMetric, NumpyOrderIndicator


class PortfolioMetrics:
    """
    设计目的:
        PortfolioMetrics用于支持投资组合相关指标的记录和计算。

    实现说明:

        账户的每日投资组合指标
        包含以下内容：收益率、成本、换手率、账户价值、现金、基准、证券价值
        对于每个交易步骤(日/分钟)，各列表示：
        - return: 策略产生的投资组合收益率(不含交易费用)
        - cost: 交易费用和滑点
        - account: 基于收盘价计算的账户总资产价值(包含现金和证券)
        - cash: 账户中的现金金额
        - bench: 基准收益率
        - value: 证券/股票/工具的总价值(不包含现金)

        更新报告
    """

    def __init__(self, freq: str = "day", benchmark_config: dict = {}) -> None:
        """
        参数
        ----------
        freq : str
            交易频率，用于更新交易柱的持有计数
        benchmark_config : dict
            基准配置，可能包含以下参数:
            - benchmark : Union[str, list, pd.Series]
                - 如果`benchmark`是pd.Series，`index`为交易日；值T表示从T-1到T的变化
                    示例:
                        print(
                            D.features(D.instruments('csi500'),
                            ['$close/Ref($close, 1)-1'])['$close/Ref($close, 1)-1'].head()
                        )
                            2017-01-04    0.011693
                            2017-01-05    0.000721
                            2017-01-06   -0.004322
                            2017-01-09    0.006874
                            2017-01-10   -0.003350
                - 如果`benchmark`是list，将使用列表中股票池的日平均变化作为'bench'
                - 如果`benchmark`是str，将使用该股票的日变化作为'bench'
                基准代码，默认为SH000300 CSI300
            - start_time : Union[str, pd.Timestamp], optional
                - 如果`benchmark`是pd.Series，此参数将被忽略
                - 否则表示基准开始时间，默认为None
            - end_time : Union[str, pd.Timestamp], optional
                - 如果`benchmark`是pd.Series，此参数将被忽略
                - 否则表示基准结束时间，默认为None
        """

        self.init_vars()
        self.init_bench(freq=freq, benchmark_config=benchmark_config)

    def init_vars(self) -> None:
        self.accounts: dict = OrderedDict()  # account position value for each trade time
        self.returns: dict = OrderedDict()  # daily return rate for each trade time
        self.total_turnovers: dict = OrderedDict()  # total turnover for each trade time
        self.turnovers: dict = OrderedDict()  # turnover for each trade time
        self.total_costs: dict = OrderedDict()  # total trade cost for each trade time
        self.costs: dict = OrderedDict()  # trade cost rate for each trade time
        self.values: dict = OrderedDict()  # value for each trade time
        self.cashes: dict = OrderedDict()
        self.benches: dict = OrderedDict()
        self.latest_pm_time: Optional[pd.TimeStamp] = None

    def init_bench(self, freq: str | None = None, benchmark_config: dict | None = None) -> None:
        if freq is not None:
            self.freq = freq
        self.benchmark_config = benchmark_config
        self.bench = self._cal_benchmark(self.benchmark_config, self.freq)

    @staticmethod
    def _cal_benchmark(benchmark_config: Optional[dict], freq: str) -> Optional[pd.Series]:
        """
        计算基准指标
        
        参数
        ----------
        benchmark_config : Optional[dict]
            基准配置
        freq : str
            交易频率
            
        返回值
        -------
        Optional[pd.Series]
            基准序列
        """
        if benchmark_config is None:
            return None
        benchmark = benchmark_config.get("benchmark", CSI300_BENCH)
        if benchmark is None:
            return None

        if isinstance(benchmark, pd.Series):
            return benchmark
        else:
            start_time = benchmark_config.get("start_time", None)
            end_time = benchmark_config.get("end_time", None)

            if freq is None:
                raise ValueError("benchmark freq can't be None!")
            _codes = benchmark if isinstance(benchmark, (list, dict)) else [benchmark]
            fields = ["$close/Ref($close,1)-1"]
            _temp_result, _ = get_higher_eq_freq_feature(_codes, fields, start_time, end_time, freq=freq)
            if len(_temp_result) == 0:
                raise ValueError(f"The benchmark {_codes} does not exist. Please provide the right benchmark")
            return (
                _temp_result.groupby(level="datetime", group_keys=False)[_temp_result.columns.tolist()[0]]
                .mean()
                .fillna(0)
            )

    def _sample_benchmark(
        self,
        bench: pd.Series,
        trade_start_time: Union[str, pd.Timestamp],
        trade_end_time: Union[str, pd.Timestamp],
    ) -> Optional[float]:
        if self.bench is None:
            return None

        def cal_change(x):
            return (x + 1).prod()

        _ret = resam_ts_data(bench, trade_start_time, trade_end_time, method=cal_change)
        return 0.0 if _ret is None else _ret - 1

    def is_empty(self) -> bool:
        return len(self.accounts) == 0

    def get_latest_date(self) -> pd.Timestamp:
        return self.latest_pm_time

    def get_latest_account_value(self) -> float:
        return self.accounts[self.latest_pm_time]

    def get_latest_total_cost(self) -> Any:
        return self.total_costs[self.latest_pm_time]

    def get_latest_total_turnover(self) -> Any:
        return self.total_turnovers[self.latest_pm_time]

    def update_portfolio_metrics_record(
        self,
        trade_start_time: Union[str, pd.Timestamp] = None,
        trade_end_time: Union[str, pd.Timestamp] = None,
        account_value: float | None = None,
        cash: float | None = None,
        return_rate: float | None = None,
        total_turnover: float | None = None,
        turnover_rate: float | None = None,
        total_cost: float | None = None,
        cost_rate: float | None = None,
        stock_value: float | None = None,
        bench_value: float | None = None,
    ) -> None:
        # check data
        if None in [
            trade_start_time,
            account_value,
            cash,
            return_rate,
            total_turnover,
            turnover_rate,
            total_cost,
            cost_rate,
            stock_value,
        ]:
            raise ValueError(
                "以下关键指标存在空值: [trade_start_time, account_value, cash, return_rate, total_turnover, turnover_rate, "
                "total_cost, cost_rate, stock_value]",
            )

        if trade_end_time is None and bench_value is None:
            raise ValueError("trade_end_time和bench_value均为空，基准指标不可用")
        elif bench_value is None:
            bench_value = self._sample_benchmark(self.bench, trade_start_time, trade_end_time)

        # update pm data
        self.accounts[trade_start_time] = account_value
        self.returns[trade_start_time] = return_rate
        self.total_turnovers[trade_start_time] = total_turnover
        self.turnovers[trade_start_time] = turnover_rate
        self.total_costs[trade_start_time] = total_cost
        self.costs[trade_start_time] = cost_rate
        self.values[trade_start_time] = stock_value
        self.cashes[trade_start_time] = cash
        self.benches[trade_start_time] = bench_value
        # update pm
        self.latest_pm_time = trade_start_time
        # finish pm update in each step

    def generate_portfolio_metrics_dataframe(self) -> pd.DataFrame:
        pm = pd.DataFrame()
        pm["account"] = pd.Series(self.accounts)
        pm["return"] = pd.Series(self.returns)
        pm["total_turnover"] = pd.Series(self.total_turnovers)
        pm["turnover"] = pd.Series(self.turnovers)
        pm["total_cost"] = pd.Series(self.total_costs)
        pm["cost"] = pd.Series(self.costs)
        pm["value"] = pd.Series(self.values)
        pm["cash"] = pd.Series(self.cashes)
        pm["bench"] = pd.Series(self.benches)
        pm.index.name = "datetime"
        return pm

    def save_portfolio_metrics(self, path: str) -> None:
        r = self.generate_portfolio_metrics_dataframe()
        r.to_csv(path)

    def load_portfolio_metrics(self, path: str) -> None:
        """从文件加载投资组合指标
        文件应包含以下列:
        columns = ['account', 'return', 'total_turnover', 'turnover', 'cost', 'total_cost', 'value', 'cash', 'bench']
        参数:
            path: str/ pathlib.Path()
        """
        with pathlib.Path(path).open("rb") as f:
            r = pd.read_csv(f, index_col=0)
        r.index = pd.DatetimeIndex(r.index)

        index = r.index
        self.init_vars()
        for trade_start_time in index:
            self.update_portfolio_metrics_record(
                trade_start_time=trade_start_time,
                account_value=r.loc[trade_start_time]["account"],
                cash=r.loc[trade_start_time]["cash"],
                return_rate=r.loc[trade_start_time]["return"],
                total_turnover=r.loc[trade_start_time]["total_turnover"],
                turnover_rate=r.loc[trade_start_time]["turnover"],
                total_cost=r.loc[trade_start_time]["total_cost"],
                cost_rate=r.loc[trade_start_time]["cost"],
                stock_value=r.loc[trade_start_time]["value"],
                bench_value=r.loc[trade_start_time]["bench"],
            )


class Indicator:
    """
    `Indicator`采用聚合方式实现。
    所有指标都是聚合计算的。
    所有指标都是针对单个股票在特定层级和特定步骤上计算的。

    | 指标         | 描述                                                         |
    |--------------+--------------------------------------------------------------|
    | amount       | 外部策略给出的*目标*交易量                                   |
    | deal_amount  | 实际成交数量                                                 |
    | inner_amount | 内部策略的总*目标*交易量                                     |
    | trade_price  | 平均成交价格                                                 |
    | trade_value  | 总交易价值                                                   |
    | trade_cost   | 总交易成本(基准价格需要考虑方向)                             |
    | trade_dir    | 交易方向                                                     |
    | ffr          | 完全成交率                                                   |
    | pa           | 价格优势                                                     |
    | pos          | 胜率                                                         |
    | base_price   | 基准价格                                                     |
    | base_volume  | 基准交易量(用于加权聚合base_price)                           |

    **注意**:
    当某步骤没有交易时，`base_price`和`base_volume`不能为NaN，否则聚合计算会得到错误结果。

    因此`base_price`不会以聚合方式计算!!
    """

    def __init__(self, order_indicator_cls: Type[BaseOrderIndicator] = NumpyOrderIndicator) -> None:
        self.order_indicator_cls = order_indicator_cls

        # 订单指标是特定步骤单个订单的度量指标
        self.order_indicator_his: dict = OrderedDict()
        self.order_indicator: BaseOrderIndicator = self.order_indicator_cls()

        # 交易指标是特定步骤所有订单的度量指标
        self.trade_indicator_his: dict = OrderedDict()
        self.trade_indicator: Dict[str, Optional[BaseSingleMetric]] = OrderedDict()

        self._trade_calendar = None

    # def reset(self, trade_calendar: TradeCalendarManager):
    def reset(self) -> None:
        self.order_indicator = self.order_indicator_cls()
        self.trade_indicator = OrderedDict()
        # self._trade_calendar = trade_calendar

    def record(self, trade_start_time: Union[str, pd.Timestamp]) -> None:
        self.order_indicator_his[trade_start_time] = self.get_order_indicator()
        self.trade_indicator_his[trade_start_time] = self.get_trade_indicator()

    def _update_order_trade_info(self, trade_info: List[Tuple[Order, float, float, float]]) -> None:
        amount = dict()
        deal_amount = dict()
        trade_price = dict()
        trade_value = dict()
        trade_cost = dict()
        trade_dir = dict()
        pa = dict()

        for order, _trade_val, _trade_cost, _trade_price in trade_info:
            amount[order.stock_id] = order.amount_delta
            deal_amount[order.stock_id] = order.deal_amount_delta
            trade_price[order.stock_id] = _trade_price
            trade_value[order.stock_id] = _trade_val * order.sign
            trade_cost[order.stock_id] = _trade_cost
            trade_dir[order.stock_id] = order.direction
            # The PA in the innermost layer is meanless
            pa[order.stock_id] = 0

        self.order_indicator.assign("amount", amount)
        self.order_indicator.assign("inner_amount", amount)
        self.order_indicator.assign("deal_amount", deal_amount)
        # 注意：trade_price和baseline price在最低层级是相同的
        self.order_indicator.assign("trade_price", trade_price)
        self.order_indicator.assign("trade_value", trade_value)
        self.order_indicator.assign("trade_cost", trade_cost)
        self.order_indicator.assign("trade_dir", trade_dir)
        self.order_indicator.assign("pa", pa)

    def _update_order_fulfill_rate(self) -> None:
        def func(deal_amount, amount):
            # deal_amount is np.nan or None when there is no inner decision. So full fill rate is 0.
            tmp_deal_amount = deal_amount.reindex(amount.index, 0)
            tmp_deal_amount = tmp_deal_amount.replace({np.nan: 0})
            return tmp_deal_amount / amount

        self.order_indicator.transfer(func, "ffr")

    def update_order_indicators(self, trade_info: List[Tuple[Order, float, float, float]]) -> None:
        self._update_order_trade_info(trade_info=trade_info)
        self._update_order_fulfill_rate()

    def _agg_order_trade_info(self, inner_order_indicators: List[BaseOrderIndicator]) -> None:
        # calculate total trade amount with each inner order indicator.
        def trade_amount_func(deal_amount, trade_price):
            return deal_amount * trade_price

        for indicator in inner_order_indicators:
            indicator.transfer(trade_amount_func, "trade_price")

        # sum inner order indicators with same metric.
        all_metric = ["inner_amount", "deal_amount", "trade_price", "trade_value", "trade_cost", "trade_dir"]
        self.order_indicator_cls.sum_all_indicators(
            self.order_indicator,
            inner_order_indicators,
            all_metric,
            fill_value=0,
        )

        def func(trade_price, deal_amount):
            # trade_price is np.nan instead of inf when deal_amount is zero.
            tmp_deal_amount = deal_amount.replace({0: np.nan})
            return trade_price / tmp_deal_amount

        self.order_indicator.transfer(func, "trade_price")

        def func_apply(trade_dir):
            return trade_dir.apply(Order.parse_dir)

        self.order_indicator.transfer(func_apply, "trade_dir")

    def _update_trade_amount(self, outer_trade_decision: BaseTradeDecision) -> None:
        # 注意：这些指标是为订单执行设计的，所以
        decision: List[Order] = cast(List[Order], outer_trade_decision.get_decision())
        if len(decision) == 0:
            self.order_indicator.assign("amount", {})
        else:
            self.order_indicator.assign("amount", {order.stock_id: order.amount_delta for order in decision})

    def _get_base_vol_pri(
        self,
        inst: str,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        direction: OrderDir,
        decision: BaseTradeDecision,
        trade_exchange: Exchange,
        pa_config: dict = {},
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        获取基础成交量和价格信息
        所有基础价格值都源自此函数
        """

        agg = pa_config.get("agg", "twap").lower()
        price = pa_config.get("price", "deal_price").lower()

        if decision.trade_range is not None:
            trade_start_time, trade_end_time = decision.trade_range.clip_time_range(
                start_time=trade_start_time,
                end_time=trade_end_time,
            )

        if price == "deal_price":
            price_s = trade_exchange.get_deal_price(
                inst,
                trade_start_time,
                trade_end_time,
                direction=direction,
                method=None,
            )
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # if there is no stock data during the time period
        if price_s is None:
            return None, None

        if isinstance(price_s, (int, float, np.number)):
            price_s = idd.SingleData(price_s, [trade_start_time])
        elif isinstance(price_s, idd.SingleData):
            pass
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # 注意：交易价格中存在一些零值，这些情况已知无意义
        # 为了与之前的逻辑保持一致，移除这些值
        # 移除零值和负值
        assert isinstance(price_s, idd.SingleData)
        price_s = price_s.loc[(price_s > 1e-08).data.astype(bool)]
        # 注意：~(price_s < 1e-08)与price_s >= 1e-8不同
        #   ~(np.nan < 1e-8) -> ~(False)  -> True

        # if price_s is empty
        if price_s.empty:
            return None, None

        assert isinstance(price_s, idd.SingleData)
        if agg == "vwap":
            volume_s = trade_exchange.get_volume(inst, trade_start_time, trade_end_time, method=None)
            if isinstance(volume_s, (int, float, np.number)):
                volume_s = idd.SingleData(volume_s, [trade_start_time])
            assert isinstance(volume_s, idd.SingleData)
            volume_s = volume_s.reindex(price_s.index)
        elif agg == "twap":
            volume_s = idd.SingleData(1, price_s.index)
        else:
            raise NotImplementedError(f"This type of input is not supported")

        assert isinstance(volume_s, idd.SingleData)
        base_volume = volume_s.sum()
        base_price = (price_s * volume_s).sum() / base_volume
        return base_price, base_volume

    def _agg_base_price(
        self,
        inner_order_indicators: List[BaseOrderIndicator],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]],
        trade_exchange: Exchange,
        pa_config: dict = {},
    ) -> None:
        """
        # 注意：!!!!
        # 重要假设!!!!!!
        # base_price的正确性依赖于使用**相同的**交易所

        参数
        ----------
        inner_order_indicators : List[BaseOrderIndicator]
            内部执行器的账户指标列表
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]],
            与inner_order_indicators对应的决策列表
        trade_exchange : Exchange
            用于获取交易价格的交易所
        pa_config : dict
            配置字典，例如
            {
                "agg": "twap",  # 可选"vwap"
                "price": "$close",  # TODO: 当前不支持此参数!!!!!
                                    # 默认使用交易所的成交价格
            }

        功能说明
        ----------
        该方法用于聚合基础价格(base_price)和基础交易量(base_volume)指标。
        1. 遍历所有内部订单指标和对应决策
        2. 对于每个股票代码，如果基础价格不存在，则调用_get_base_vol_pri获取
        3. 计算加权平均基础价格和总基础交易量

        返回值
        ----------
        无返回值，但会更新order_indicator中的以下指标：
        - base_volume: 基础交易量
        - base_price: 加权平均基础价格
        """

        # TODO: 我认为这里还有优化空间
        trade_dir = self.order_indicator.get_index_data("trade_dir")
        if len(trade_dir) > 0:
            bp_all, bv_all = [], []
            # <step, inst, (base_volume | base_price)>
            for oi, (dec, start, end) in zip(inner_order_indicators, decision_list):
                bp_s = oi.get_index_data("base_price").reindex(trade_dir.index)
                bv_s = oi.get_index_data("base_volume").reindex(trade_dir.index)

                bp_new, bv_new = {}, {}
                for pr, v, (inst, direction) in zip(bp_s.data, bv_s.data, zip(trade_dir.index, trade_dir.data)):
                    if np.isnan(pr):
                        bp_tmp, bv_tmp = self._get_base_vol_pri(
                            inst,
                            start,
                            end,
                            decision=dec,
                            direction=direction,
                            trade_exchange=trade_exchange,
                            pa_config=pa_config,
                        )
                        if (bp_tmp is not None) and (bv_tmp is not None):
                            bp_new[inst], bv_new[inst] = bp_tmp, bv_tmp
                    else:
                        bp_new[inst], bv_new[inst] = pr, v

                bp_new = idd.SingleData(bp_new)
                bv_new = idd.SingleData(bv_new)
                bp_all.append(bp_new)
                bv_all.append(bv_new)
            bp_all_multi_data = idd.concat(bp_all, axis=1)
            bv_all_multi_data = idd.concat(bv_all, axis=1)

            base_volume = bv_all_multi_data.sum(axis=1)
            self.order_indicator.assign("base_volume", base_volume.to_dict())
            self.order_indicator.assign(
                "base_price",
                ((bp_all_multi_data * bv_all_multi_data).sum(axis=1) / base_volume).to_dict(),
            )

    def _agg_order_price_advantage(self) -> None:
        def if_empty_func(trade_price):
            return trade_price.empty

        if_empty = self.order_indicator.transfer(if_empty_func)
        if not if_empty:

            def func(trade_dir, trade_price, base_price):
                sign = 1 - trade_dir * 2
                return sign * (trade_price / base_price - 1)

            self.order_indicator.transfer(func, "pa")
        else:
            self.order_indicator.assign("pa", {})

    def agg_order_indicators(
        self,
        inner_order_indicators: List[BaseOrderIndicator],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]],
        outer_trade_decision: BaseTradeDecision,
        trade_exchange: Exchange,
        indicator_config: dict = {},
    ) -> None:
        self._agg_order_trade_info(inner_order_indicators)
        self._update_trade_amount(outer_trade_decision)
        self._update_order_fulfill_rate()
        pa_config = indicator_config.get("pa_config", {})
        self._agg_base_price(inner_order_indicators, decision_list, trade_exchange, pa_config=pa_config)  # TODO
        self._agg_order_price_advantage()

    def _cal_trade_fulfill_rate(self, method: str = "mean") -> Optional[BaseSingleMetric]:
        if method == "mean":
            return self.order_indicator.transfer(
                lambda ffr: ffr.mean(),
            )
        elif method == "amount_weighted":
            return self.order_indicator.transfer(
                lambda ffr, deal_amount: (ffr * deal_amount.abs()).sum() / (deal_amount.abs().sum()),
            )
        elif method == "value_weighted":
            return self.order_indicator.transfer(
                lambda ffr, trade_value: (ffr * trade_value.abs()).sum() / (trade_value.abs().sum()),
            )
        else:
            raise ValueError(f"method {method} is not supported!")

    def _cal_trade_price_advantage(self, method: str = "mean") -> Optional[BaseSingleMetric]:
        if method == "mean":
            return self.order_indicator.transfer(lambda pa: pa.mean())
        elif method == "amount_weighted":
            return self.order_indicator.transfer(
                lambda pa, deal_amount: (pa * deal_amount.abs()).sum() / (deal_amount.abs().sum()),
            )
        elif method == "value_weighted":
            return self.order_indicator.transfer(
                lambda pa, trade_value: (pa * trade_value.abs()).sum() / (trade_value.abs().sum()),
            )
        else:
            raise ValueError(f"method {method} is not supported!")

    def _cal_trade_positive_rate(self) -> Optional[BaseSingleMetric]:
        def func(pa):
            return (pa > 0).sum() / pa.count()

        return self.order_indicator.transfer(func)

    def _cal_deal_amount(self) -> Optional[BaseSingleMetric]:
        def func(deal_amount):
            return deal_amount.abs().sum()

        return self.order_indicator.transfer(func)

    def _cal_trade_value(self) -> Optional[BaseSingleMetric]:
        def func(trade_value):
            return trade_value.abs().sum()

        return self.order_indicator.transfer(func)

    def _cal_trade_order_count(self) -> Optional[BaseSingleMetric]:
        def func(amount):
            return amount.count()

        return self.order_indicator.transfer(func)

    def cal_trade_indicators(
        self,
        trade_start_time: Union[str, pd.Timestamp],
        freq: str,
        indicator_config: dict = {},
    ) -> None:
        show_indicator = indicator_config.get("show_indicator", False)
        ffr_config = indicator_config.get("ffr_config", {})
        pa_config = indicator_config.get("pa_config", {})
        fulfill_rate = self._cal_trade_fulfill_rate(method=ffr_config.get("weight_method", "mean"))
        price_advantage = self._cal_trade_price_advantage(method=pa_config.get("weight_method", "mean"))
        positive_rate = self._cal_trade_positive_rate()
        deal_amount = self._cal_deal_amount()
        trade_value = self._cal_trade_value()
        order_count = self._cal_trade_order_count()
        self.trade_indicator["ffr"] = fulfill_rate
        self.trade_indicator["pa"] = price_advantage
        self.trade_indicator["pos"] = positive_rate
        self.trade_indicator["deal_amount"] = deal_amount
        self.trade_indicator["value"] = trade_value
        self.trade_indicator["count"] = order_count
        if show_indicator:
            print(
                "[Indicator({}) {}]: FFR: {}, PA: {}, POS: {}".format(
                    freq,
                    (
                        trade_start_time
                        if isinstance(trade_start_time, str)
                        else trade_start_time.strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    fulfill_rate,
                    price_advantage,
                    positive_rate,
                ),
            )

    def get_order_indicator(self, raw: bool = True) -> Union[BaseOrderIndicator, Dict[Text, pd.Series]]:
        return self.order_indicator if raw else self.order_indicator.to_series()

    def get_trade_indicator(self) -> Dict[str, Optional[BaseSingleMetric]]:
        return self.trade_indicator

    def generate_trade_indicators_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.trade_indicator_his, orient="index")
