# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import warnings
from typing import Union, Literal

from ..log import get_module_logger
from ..utils import get_date_range
from ..utils.resam import Freq
from ..strategy.base import BaseStrategy
from ..backtest import get_exchange, position, backtest as backtest_func, executor as _executor


from ..data import D
from ..config import C
from ..data.dataset.utils import get_level_index


logger = get_module_logger("Evaluate")


def risk_analysis(r, N: int = None, freq: str = "day", mode: Literal["sum", "product"] = "sum"):
    """风险分析
    注意：
    年化收益率的计算方式与年化收益率的定义有所不同。
    这是有意为之的设计实现。
    Qlib尝试通过求和而非乘积来累积收益，以避免累积曲线呈指数型扭曲。
    Qlib中所有年化收益率的计算均遵循此原则。

    参数
    ----------
    r : pandas.Series
        日收益率序列。
    N: int
        信息比率年化的缩放因子（日度：252，周度：50，月度：12），`N`和`freq`至少需存在一个
    freq: str
        用于计算缩放因子的分析频率，`N`和`freq`至少需存在一个
    mode: Literal["sum", "product"]
        收益累积方式：
        - "sum": 算术累积（线性收益）。
        - "product": 几何累积（复利收益）。
    """

    def cal_risk_analysis_scaler(freq):
        _count, _freq = Freq.parse(freq)
        _freq_scaler = {
            Freq.NORM_FREQ_MINUTE: 240 * 238,
            Freq.NORM_FREQ_DAY: 238,
            Freq.NORM_FREQ_WEEK: 50,
            Freq.NORM_FREQ_MONTH: 12,
        }
        return _freq_scaler[_freq] / _count

    if N is None and freq is None:
        raise ValueError("at least one of `N` and `freq` should exist")
    if N is not None and freq is not None:
        warnings.warn("risk_analysis freq will be ignored")
    if N is None:
        N = cal_risk_analysis_scaler(freq)

    if mode == "sum":
        mean = r.mean()
        std = r.std(ddof=1)
        annualized_return = mean * N
        max_drawdown = (r.cumsum() - r.cumsum().cummax()).min()
    elif mode == "product":
        cumulative_curve = (1 + r).cumprod()
        # geometric mean (compound annual growth rate)
        mean = cumulative_curve.iloc[-1] ** (1 / len(r)) - 1
        # volatility of log returns
        std = np.log(1 + r).std(ddof=1)

        cumulative_return = cumulative_curve.iloc[-1] - 1
        annualized_return = (1 + cumulative_return) ** (N / len(r)) - 1
        # max percentage drawdown from peak cumulative product
        max_drawdown = (cumulative_curve / cumulative_curve.cummax() - 1).min()
    else:
        raise ValueError(f"risk_analysis accumulation mode {mode} is not supported. Expected `sum` or `product`.")

    information_ratio = mean / std * np.sqrt(N)
    data = {
        "mean": mean,
        "std": std,
        "annualized_return": annualized_return,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
    }
    res = pd.Series(data).to_frame("risk")
    return res


def indicator_analysis(df, method="mean"):
    """分析交易的统计时间序列指标

    参数
    ----------
    df : pandas.DataFrame
        列：如 ['pa', 'pos', 'ffr', 'deal_amount', 'value']。
            必需字段：
                - 'pa' 是交易指标中的价格优势
                - 'pos' 是交易指标中的正比率
                - 'ffr' 是交易指标中的成交率
            可选字段：
                - 'deal_amount' 是总成交量，仅在 method 为 'amount_weighted' 时需要
                - 'value' 是总交易价值，仅在 method 为 'value_weighted' 时需要

        索引：Index(datetime)
    method : str, 可选
        pa/ffr 的统计方法，默认为 "mean"

        - 如果 method 为 'mean'，计算每个交易指标的平均统计值
        - 如果 method 为 'amount_weighted'，计算每个交易指标的成交量加权平均统计值
        - 如果 method 为 'value_weighted'，计算每个交易指标的价值加权平均统计值

        注意：pos 的统计方法始终为 "mean"

    返回
    -------
    pd.DataFrame
        每个交易指标的统计值
    """
    weights_dict = {
        "mean": df["count"],
        "amount_weighted": df["deal_amount"].abs(),
        "value_weighted": df["value"].abs(),
    }
    if method not in weights_dict:
        raise ValueError(f"indicator_analysis method {method} is not supported!")

    # statistic pa/ffr indicator
    indicators_df = df[["ffr", "pa"]]
    weights = weights_dict.get(method)
    res = indicators_df.mul(weights, axis=0).sum() / weights.sum()

    # statistic pos
    weights = weights_dict.get("mean")
    res.loc["pos"] = df["pos"].mul(weights).sum() / weights.sum()
    res = res.to_frame("value")
    return res


# This is the API for compatibility for legacy code
def backtest_daily(
    start_time: Union[str, pd.Timestamp],
    end_time: Union[str, pd.Timestamp],
    strategy: Union[str, dict, BaseStrategy],
    executor: Union[str, dict, _executor.BaseExecutor] = None,
    account: Union[float, int, position.Position] = 1e8,
    benchmark: str = "SH000300",
    exchange_kwargs: dict = None,
    pos_type: str = "Position",
):
    """初始化策略和执行器，然后执行日频回测

    参数
    ----------
    start_time : Union[str, pd.Timestamp]
        回测的闭区间开始时间
        **注意**：这将应用于最外层执行器的日历。
    end_time : Union[str, pd.Timestamp]
        回测的闭区间结束时间
        **注意**：这将应用于最外层执行器的日历。
        例如：Executor[day](Executor[1min])，设置 `end_time == 20XX0301` 将包含 20XX0301 这一天的所有分钟
    strategy : Union[str, dict, BaseStrategy]
        用于初始化最外层投资组合策略。更多信息请参考 init_instance_by_config 的文档。

        例如：

        .. code-block:: python

            # 字典形式
            strategy = {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": (model, dataset),
                    "topk": 50,
                    "n_drop": 5,
                },
            }
            # BaseStrategy形式
            pred_score = pd.read_pickle("score.pkl")["score"]
            STRATEGY_CONFIG = {
                "topk": 50,
                "n_drop": 5,
                "signal": pred_score,
            }
            strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
            # 字符串形式示例：
            # 1) 指定pickle对象
            #     - 路径形如 'file:///<path to pickle file>/obj.pkl'
            # 2) 指定类名
            #     - "ClassName": 将使用 getattr(module, "ClassName")() 
            # 3) 指定带类名的模块路径
            #     - "a.b.c.ClassName" 将使用 getattr(<a.b.c.module>, "ClassName")() 

    executor : Union[str, dict, BaseExecutor]
        用于初始化最外层执行器。
    benchmark: str
        用于报告的基准。
    account : Union[float, int, Position]
        描述如何创建账户的信息

        对于 `float` 或 `int`：

            仅使用初始现金创建账户

        对于 `Position`：

            使用持仓创建账户
    exchange_kwargs : dict
        用于初始化交易所的参数
        例如：

        .. code-block:: python

            exchange_kwargs = {
                "freq": freq,
                "limit_threshold": None, # limit_threshold 为 None 时，使用 C.limit_threshold
                "deal_price": None, # deal_price 为 None 时，使用 C.deal_price
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }

    pos_type : str
        持仓类型。

    返回
    -------
    report_normal: pd.DataFrame
        回测报告
    positions_normal: pd.DataFrame
        回测持仓

    """
    freq = "day"
    if executor is None:
        executor_config = {
            "time_per_step": freq,
            "generate_portfolio_metrics": True,
        }
        executor = _executor.SimulatorExecutor(**executor_config)
    _exchange_kwargs = {
        "freq": freq,
        "limit_threshold": None,
        "deal_price": None,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }
    if exchange_kwargs is not None:
        _exchange_kwargs.update(exchange_kwargs)

    portfolio_metric_dict, indicator_dict = backtest_func(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor,
        account=account,
        benchmark=benchmark,
        exchange_kwargs=_exchange_kwargs,
        pos_type=pos_type,
    )
    analysis_freq = "{0}{1}".format(*Freq.parse(freq))

    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    return report_normal, positions_normal


def long_short_backtest(
    pred,
    topk=50,
    deal_price=None,
    shift=1,
    open_cost=0,
    close_cost=0,
    trade_unit=None,
    limit_threshold=None,
    min_cost=5,
    subscribe_fields=[],
    extract_codes=False,
):
    """
    多空策略的回测

    :param pred:        在 `T` 日产生的交易信号。
    :param topk:       做空和做多的前topk只证券。
    :param deal_price:  交易价格。
    :param shift:       是否将预测向后移动一天。如果shift==1，交易日将是T+1。
    :param open_cost:   开仓交易成本。
    :param close_cost:  平仓交易成本。
    :param trade_unit:  中国A股为100。
    :param limit_threshold: 涨跌停限制，例如0.1（10%），多空使用相同的限制。
    :param min_cost:    最小交易成本。
    :param subscribe_fields: 订阅字段。
    :param extract_codes:  布尔值。
                       是否将从pred中提取的代码传递给交易所。
                       注意：在离线qlib中这样会更快。
    :return:            回测结果，以字典形式表示。
                        { "long": 多头超额收益,
                        "short": 空头超额收益,
                        "long_short": 多空组合收益}
    """
    if get_level_index(pred, level="datetime") == 1:
        pred = pred.swaplevel().sort_index()

    if trade_unit is None:
        trade_unit = C.trade_unit
    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if deal_price is None:
        deal_price = C.deal_price
    if deal_price[0] != "$":
        deal_price = "$" + deal_price

    subscribe_fields = subscribe_fields.copy()
    profit_str = f"Ref({deal_price}, -1)/{deal_price} - 1"
    subscribe_fields.append(profit_str)

    trade_exchange = get_exchange(
        pred=pred,
        deal_price=deal_price,
        subscribe_fields=subscribe_fields,
        limit_threshold=limit_threshold,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
        trade_unit=trade_unit,
        extract_codes=extract_codes,
        shift=shift,
    )

    _pred_dates = pred.index.get_level_values(level="datetime")
    predict_dates = D.calendar(start_time=_pred_dates.min(), end_time=_pred_dates.max())
    trade_dates = np.append(predict_dates[shift:], get_date_range(predict_dates[-1], left_shift=1, right_shift=shift))

    long_returns = {}
    short_returns = {}
    ls_returns = {}

    for pdate, date in zip(predict_dates, trade_dates):
        score = pred.loc(axis=0)[pdate, :]
        score = score.reset_index().sort_values(by="score", ascending=False)

        long_stocks = list(score.iloc[:topk]["instrument"])
        short_stocks = list(score.iloc[-topk:]["instrument"])

        score = score.set_index(["datetime", "instrument"]).sort_index()

        long_profit = []
        short_profit = []
        all_profit = []

        for stock in long_stocks:
            if not trade_exchange.is_stock_tradable(stock_id=stock, trade_date=date):
                continue
            profit = trade_exchange.get_quote_info(stock_id=stock, start_time=date, end_time=date, field=profit_str)
            if np.isnan(profit):
                long_profit.append(0)
            else:
                long_profit.append(profit)

        for stock in short_stocks:
            if not trade_exchange.is_stock_tradable(stock_id=stock, trade_date=date):
                continue
            profit = trade_exchange.get_quote_info(stock_id=stock, start_time=date, end_time=date, field=profit_str)
            if np.isnan(profit):
                short_profit.append(0)
            else:
                short_profit.append(profit * -1)

        for stock in list(score.loc(axis=0)[pdate, :].index.get_level_values(level=0)):
            # exclude the suspend stock
            if trade_exchange.check_stock_suspended(stock_id=stock, trade_date=date):
                continue
            profit = trade_exchange.get_quote_info(stock_id=stock, start_time=date, end_time=date, field=profit_str)
            if np.isnan(profit):
                all_profit.append(0)
            else:
                all_profit.append(profit)

        long_returns[date] = np.mean(long_profit) - np.mean(all_profit)
        short_returns[date] = np.mean(short_profit) + np.mean(all_profit)
        ls_returns[date] = np.mean(short_profit) + np.mean(long_profit)

    return dict(
        zip(
            ["long", "short", "long_short"],
            map(pd.Series, [long_returns, short_returns, ls_returns]),
        )
    )


def t_run():
    pred_FN = "./check_pred.csv"
    pred: pd.DataFrame = pd.read_csv(pred_FN)
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred = pred.set_index([pred.columns[0], pred.columns[1]])
    pred = pred.iloc[:9000]
    strategy_config = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred,
    }
    report_df, positions = backtest_daily(start_time="2017-01-01", end_time="2020-08-01", strategy=strategy_config)
    print(report_df.head())
    print(positions.keys())
    print(positions[list(positions.keys())[0]])
    return 0


if __name__ == "__main__":
    t_run()
