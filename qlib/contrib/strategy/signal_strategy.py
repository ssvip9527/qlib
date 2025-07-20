# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union
from abc import ABC

from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        参数
        -----------
        signal :
            描述信号的信息。请参考`qlib.backtest.signal.create_signal_from`的文档
            策略的决策将基于给定的信号
        risk_degree : float
            总价值的持仓百分比
        trade_exchange : Exchange
            提供市场信息的交易所，用于处理订单和生成报告
            - 如果`trade_exchange`为None，self.trade_exchange将通过common_infra设置
            - 允许在不同的执行中使用不同的trade_exchange
            - 例如：
                - 在日频执行中，日频和分钟频交易所都可用，但推荐使用日频交易所，因为它运行更快
                - 在分钟频执行中，日频交易所不可用，只推荐使用分钟频交易所
        """
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)

        self.risk_degree = risk_degree

        # 这是为了兼容旧版本的qlib任务配置
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset`已弃用；请使用`signal`", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """获取风险度
        返回你将用于投资的总价值比例
        动态风险度将导致市场择时
        """
        # 默认使用总价值的95%
        return self.risk_degree


class TopkDropoutStrategy(BaseSignalStrategy):
    # TODO:
    # 1. 支持从决策中获取range_limit结果
    # 2. 支持修改外部交易决策
    # 3. 支持检查交易决策的可用性
    # 4. 将forbid_all_trade_at_limit设为false重新生成结果，并将其默认值改为false，因为这与现实一致
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        """
        参数
        -----------
        topk : int
            投资组合中的股票数量
        n_drop : int
            每个交易日要替换的股票数量
        method_sell : str
            卖出方法，random/bottom
        method_buy : str
            买入方法，random/top
        hold_thresh : int
            最小持有天数
            在卖出股票前，会检查current.get_stock_count(order.stock_id) >= self.hold_thresh
        only_tradable : bool
            策略是否只考虑可交易股票

            如果only_tradable为True:

                策略将根据股票的可交易状态做出决策，避免买卖不可交易股票

            否则:

                策略将在不考虑股票可交易状态的情况下做出买卖决策
        forbid_all_trade_at_limit : bool
            是否在达到涨跌停时禁止所有交易

            如果forbid_all_trade_at_limit为True:

                当价格达到涨跌停时，策略不会进行任何交易，即使在现实中允许在涨停卖出和在跌停买入

            否则:

                策略将在涨停卖出和在跌停买入
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def generate_trade_decision(self, execute_result=None):
        """生成交易决策

        根据信号生成交易决策，包括选股和调仓逻辑。

        :param execute_result: 执行结果，默认为None
        :type execute_result: Any
        :return: 交易决策对象
        :rtype: TradeDecisionWO
        """
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision without the stock tradable info
            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        # last position (sorted by score)
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy
        buy = today[: len(sell) + self.topk - len(last)]
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                # check hold limit
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
                # sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # update cash
                    cash += trade_val - trade_cost
        # buy new stock
        # note the current has been changed
        # current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            # buy order
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,  # 1 for buy
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)


class WeightStrategyBase(BaseSignalStrategy):
    # TODO:
    # 1. 支持从决策中获取range_limit结果
    # 2. 支持修改外部交易决策
    # 3. 支持检查交易决策的可用性
    def __init__(
        self,
        *,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        signal :
            描述信号的信息。请参考`qlib.backtest.signal.create_signal_from`的文档
            策略的决策将基于给定的信号
        trade_exchange : Exchange
            提供市场信息的交易所，用于处理订单和生成报告

            - 如果`trade_exchange`为None，self.trade_exchange将通过common_infra设置
            - 允许在不同的执行中使用不同的trade_exchange
            - 例如：

                - 在日频执行中，日频和分钟频交易所都可用，但推荐使用日频交易所，因为它运行更快
                - 在分钟频执行中，日频交易所不可用，只推荐使用分钟频交易所
        """
        super().__init__(**kwargs)

        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        从分数生成目标持仓位置。现金不考虑在持仓中

        参数
        -----------
        score : pd.Series
            当前交易日的预测分数，索引为stock_id，包含'score'列
        current : Position()
            当前持仓
        trade_start_time: pd.Timestamp
        trade_end_time: pd.Timestamp
        """
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # 使用generate_target_weight_position()和generate_order_list_from_target_weight_position()生成订单列表

        # 获取已完成的交易步骤数，trade_step可以是[0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # 避免InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time
        )
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(trade_step),
            target_weight_position=target_weight_position,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        return TradeDecisionWO(order_list, self)


class EnhancedIndexingStrategy(WeightStrategyBase):
    """增强指数策略

    增强指数策略结合了主动管理和被动管理的艺术，
    旨在控制风险敞口（又称跟踪误差）的同时超越基准指数（如标普500）的投资组合回报。

    用户需要准备如下的风险模型数据：

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # 可选

    风险模型数据可以从风险数据提供商处获取。你也可以使用
    `qlib.model.riskmodel.structured.StructuredCovEstimator`来准备这些数据。

    参数:
        riskmodel_path (str): 风险模型路径
        name_mapping (dict): 替代文件名
    """

    FACTOR_EXP_NAME = "factor_exp.pkl"
    FACTOR_COV_NAME = "factor_cov.pkl"
    SPECIFIC_RISK_NAME = "specific_risk.pkl"
    BLACKLIST_NAME = "blacklist.pkl"

    def __init__(
        self,
        *,
        riskmodel_root,
        market="csi500",
        turn_limit=None,
        name_mapping={},
        optimizer_kwargs={},
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = get_module_logger("EnhancedIndexingStrategy")

        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit

        self.factor_exp_path = name_mapping.get("factor_exp", self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get("factor_cov", self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get("specific_risk", self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get("blacklist", self.BLACKLIST_NAME)

        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)

        self.verbose = verbose

        self._riskdata_cache = {}

    def get_risk_data(self, date):
        """
        获取指定日期的风险数据
        
        参数:
            date: 日期
            
        返回:
            因子暴露矩阵、因子协方差矩阵、特定风险向量、股票列表、黑名单
        """
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]

        root = self.riskmodel_root + "/" + date.strftime("%Y%m%d")
        if not os.path.exists(root):
            return None

        factor_exp = load_dataset(root + "/" + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + "/" + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + "/" + self.specific_risk_path, index_col=[0])

        if not factor_exp.index.equals(specific_risk.index):
            # 注意：对于缺少特定风险的股票，我们总是假设它具有最高的波动性
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())

        universe = factor_exp.index.tolist()

        blacklist = []
        if os.path.exists(root + "/" + self.blacklist_path):
            blacklist = load_dataset(root + "/" + self.blacklist_path).index.tolist()

        self._riskdata_cache[date] = factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist

        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        生成目标持仓权重
        
        参数:
            score: 预测分数
            current: 当前持仓
            trade_start_time: 交易开始时间
            trade_end_time: 交易结束时间
            
        返回:
            目标持仓权重字典
        """
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)  # 上一个交易日

        # 加载风险数据
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f"没有找到{pre_date:%Y-%m-%d}的风险数据，跳过优化")
            return None
        factor_exp, factor_cov, specific_risk, universe, blacklist = outs

        # 转换分数
        # 注意：对于缺少分数的股票，我们总是假设它们具有最低的分数
        score = score.reindex(universe).fillna(score.min()).values

        # 获取当前权重
        # 注意：如果股票不在universe中，其当前权重将为零
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), "当前权重包含负值"
        cur_weight = cur_weight / self.get_risk_degree(trade_date)  # 权重总和应为risk_degree
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f"先前总持仓超过风险度(当前: {cur_weight.sum()})")

        # 加载基准权重
        bench_weight = D.features(
            D.instruments("all"), [f"${self.market}_weight"], start_time=pre_date, end_time=pre_date
        ).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level="datetime")
        bench_weight = bench_weight.reindex(universe).fillna(0).values

        # 检查股票是否可交易
        # 注意：目前我们使用前一天的成交量来检查是否可交易
        tradable = D.features(D.instruments("all"), ["$volume"], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level="datetime")
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable

        # 强制卖出标记
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)

        # 优化
        weight = self.optimizer(
            r=score,
            F=factor_exp,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=cur_weight,
            wb=bench_weight,
            mfh=mask_force_hold,
            mfs=mask_force_sell,
        )

        target_weight_position = {stock: weight for stock, weight in zip(universe, weight) if weight > 0}

        if self.verbose:
            self.logger.info("交易日期: {:%Y-%m-%d}".format(trade_date))
            self.logger.info("持仓股票数量: {}".format(len(target_weight_position)))
            self.logger.info("总持仓权重: {:.6f}".format(weight.sum()))

        return target_weight_position
