# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from ..data import D

from collections import OrderedDict


def _get_position_value_from_df(evaluate_date, position, close_data_df):
    """根据已有的收盘价数据DataFrame获取持仓价值
    close_data_df:
        pd.DataFrame类型
        多层索引
        close_data_df['$close'][stock_id][evaluate_date]表示(股票ID, 评估日期)的收盘价
    position:
        与get_position_value()中的position参数相同
    """
    value = 0
    for stock_id, report in position.items():
        if stock_id != "cash":
            value += report["amount"] * close_data_df["$close"][stock_id][evaluate_date]
            # value += report['amount'] * report['price']
    if "cash" in position:
        value += position["cash"]
    return value


def get_position_value(evaluate_date, position):
    """收盘价乘以数量的总和

    获取持仓的价值

    使用收盘价

        positions示例:
        {
            Timestamp('2016-01-05 00:00:00'):
            {
                'SH600022':
                {
                    'amount':100.00,
                    'price':12.00
                },

                'cash':100000.0
            }
        }

    表示在'2016-01-05'持有100股'SH600022'和100000元人民币现金
    """
    # load close price for position
    # position should also consider cash
    instruments = list(position.keys())
    instruments = list(set(instruments) - {"cash"})  # filter 'cash'
    fields = ["$close"]
    close_data_df = D.features(
        instruments,
        fields,
        start_time=evaluate_date,
        end_time=evaluate_date,
        freq="day",
        disk_cache=0,
    )
    value = _get_position_value_from_df(evaluate_date, position, close_data_df)
    return value


def get_position_list_value(positions):
    # generate instrument list and date for whole poitions
    instruments = set()
    for day, position in positions.items():
        instruments.update(position.keys())
    instruments = list(set(instruments) - {"cash"})  # filter 'cash'
    instruments.sort()
    day_list = list(positions.keys())
    day_list.sort()
    start_date, end_date = day_list[0], day_list[-1]
    # load data
    fields = ["$close"]
    close_data_df = D.features(
        instruments,
        fields,
        start_time=start_date,
        end_time=end_date,
        freq="day",
        disk_cache=0,
    )
    # generate value
    # return dict for time:position_value
    value_dict = OrderedDict()
    for day, position in positions.items():
        value = _get_position_value_from_df(evaluate_date=day, position=position, close_data_df=close_data_df)
        value_dict[day] = value
    return value_dict


def get_daily_return_series_from_positions(positions, init_asset_value):
    """参数说明
    从持仓视图生成日收益率序列
    positions: 策略生成的持仓数据
    init_asset_value: 初始资产价值
    return: 日收益率的pd.Series对象，return_series[date] = 日收益率
    """
    value_dict = get_position_list_value(positions)
    value_series = pd.Series(value_dict)
    value_series = value_series.sort_index()  # check date
    return_series = value_series.pct_change()
    return_series[value_series.index[0]] = (
        value_series[value_series.index[0]] / init_asset_value - 1
    )  # update daily return for the first date
    return return_series


def get_annual_return_from_positions(positions, init_asset_value):
    """年化收益率

    p_r = (p_end / p_start)^{(250/n)} - 1

    p_r     年化收益率
    p_end   最终价值
    p_start 初始价值
    n       回测天数

    """
    date_range_list = sorted(list(positions.keys()))
    end_time = date_range_list[-1]
    p_end = get_position_value(end_time, positions[end_time])
    p_start = init_asset_value
    n_period = len(date_range_list)
    annual = pow((p_end / p_start), (250 / n_period)) - 1

    return annual


def get_annaul_return_from_return_series(r, method="ci"):
    """基于日收益率序列的风险分析

    参数
    ----------
    r : pandas.Series
        日收益率序列
    method : str
        利息计算方法，ci(复利)/si(单利)
    """
    mean = r.mean()
    annual = (1 + mean) ** 250 - 1 if method == "ci" else mean * 250

    return annual


def get_sharpe_ratio_from_return_series(r, risk_free_rate=0.00, method="ci"):
    """风险分析

    参数
    ----------
    r : pandas.Series
        日收益率序列
    method : str
        利息计算方法，ci(复利)/si(单利)
    risk_free_rate : float
        无风险利率，默认为0.00，可设置为0.03等
    """
    std = r.std(ddof=1)
    annual = get_annaul_return_from_return_series(r, method=method)
    sharpe = (annual - risk_free_rate) / std / np.sqrt(250)

    return sharpe


def get_max_drawdown_from_series(r):
    """基于资产价值的风险分析

    使用累积乘积方法

    参数
    ----------
    r : pandas.Series
        日收益率序列
    """
    # mdd = ((r.cumsum() - r.cumsum().cummax()) / (1 + r.cumsum().cummax())).min()

    mdd = (((1 + r).cumprod() - (1 + r).cumprod().cummax()) / ((1 + r).cumprod().cummax())).min()

    return mdd


def get_turnover_rate():
    # in backtest
    pass


def get_beta(r, b):
    """Risk Analysis  beta

    Parameters
    ----------
    r : pandas.Series
        daily return series of strategy
    b : pandas.Series
        daily return series of baseline
    """
    cov_r_b = np.cov(r, b)
    var_b = np.var(b)
    return cov_r_b / var_b


def get_alpha(r, b, risk_free_rate=0.03):
    beta = get_beta(r, b)
    annaul_r = get_annaul_return_from_return_series(r)
    annaul_b = get_annaul_return_from_return_series(b)

    alpha = annaul_r - risk_free_rate - beta * (annaul_b - risk_free_rate)

    return alpha


def get_volatility_from_series(r):
    return r.std(ddof=1)


def get_rank_ic(a, b):
    """Rank IC

    Parameters
    ----------
    r : pandas.Series
        daily score series of feature
    b : pandas.Series
        daily return series

    """
    return spearmanr(a, b).correlation


def get_normal_ic(a, b):
    return pearsonr(a, b)[0]
