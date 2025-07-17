# 版权所有 (c) Microsoft Corporation
# MIT许可证授权
"""
该模块维护状态不佳
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import C
from ..data import D
from .position import Position


def get_benchmark_weight(
    bench,
    start_date=None,
    end_date=None,
    path=None,
    freq="day",
):
    """获取基准指数的股票权重分布

    参数
    ----------
    bench : str
        基准指数代码
    start_date : str, optional
        开始日期
    end_date : str, optional
        结束日期
    path : str, optional
        权重数据文件路径
    freq : str, optional
        数据频率，默认为'day'

    返回值
    ----------
    pd.DataFrame
        基准指数的权重分布数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 每个单元格表示该股票在基准中的权重

    注意事项
    ----------
    - 权重数据存储方式需要改进
    - 基准指数与文件名中的索引可能不一致
    """
    if not path:
        path = Path(C.dpm.get_data_uri(freq)).expanduser() / "raw" / "AIndexMembers" / "weights.csv"
    # TODO: the storage of weights should be implemented in a more elegent way
    # TODO: The benchmark is not consistent with the filename in instruments.
    bench_weight_df = pd.read_csv(path, usecols=["code", "date", "index", "weight"])
    bench_weight_df = bench_weight_df[bench_weight_df["index"] == bench]
    bench_weight_df["date"] = pd.to_datetime(bench_weight_df["date"])
    if start_date is not None:
        bench_weight_df = bench_weight_df[bench_weight_df.date >= start_date]
    if end_date is not None:
        bench_weight_df = bench_weight_df[bench_weight_df.date <= end_date]
    bench_stock_weight = bench_weight_df.pivot_table(index="date", columns="code", values="weight") / 100.0
    return bench_stock_weight


def get_stock_weight_df(positions):
    """
    获取投资组合的股票权重分布

    参数
    ----------
    positions : dict
        回测结果中的持仓字典
        
    返回值
    ----------
    pd.DataFrame
        投资组合的权重分布数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 每个单元格表示该股票在投资组合中的权重
    """
    stock_weight = []
    index = []
    for date in sorted(positions.keys()):
        pos = positions[date]
        if isinstance(pos, dict):
            pos = Position(position_dict=pos)
        index.append(date)
        stock_weight.append(pos.get_stock_weight_dict(only_stock=True))
    return pd.DataFrame(stock_weight, index=index)


def decompose_portofolio_weight(stock_weight_df, stock_group_df):
    """
    分解投资组合权重到各分组

    参数
    ----------
    stock_weight_df : pd.DataFrame
        描述投资组合权重的数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 示例:
            code        SH600004  SH600006  SH600017  SH600022  SH600026  SH600037  \
            date
            2016-01-05  0.001543  0.001570  0.002732  0.001320  0.003000       NaN
            2016-01-06  0.001538  0.001569  0.002770  0.001417  0.002945       NaN
    stock_group_df : pd.DataFrame
        描述股票分组的数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 单元格值为分组ID
        - 示例(行业分组):
            instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008  \
            datetime
            2016-01-05  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
            2016-01-06  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0

    返回值
    ----------
    tuple
        包含两个字典的元组:
        1. group_weight: 分组权重字典
           - 键: 分组ID
           - 值: 描述该分组权重的Series
        2. stock_weight_in_group: 组内股票权重字典
           - 键: 分组ID
           - 值: 描述组内各股票权重的DataFrame
    """
    all_group = np.unique(stock_group_df.values.flatten())
    all_group = all_group[~np.isnan(all_group)]

    group_weight = {}
    stock_weight_in_group = {}
    for group_key in all_group:
        group_mask = stock_group_df == group_key
        group_weight[group_key] = stock_weight_df[group_mask].sum(axis=1)
        stock_weight_in_group[group_key] = stock_weight_df[group_mask].divide(group_weight[group_key], axis=0)
    return group_weight, stock_weight_in_group


def decompose_portofolio(stock_weight_df, stock_group_df, stock_ret_df):
    """
    分解投资组合到分组权重和分组收益

    参数
    ----------
    stock_weight_df : pd.DataFrame
        描述投资组合权重的数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 示例:
            code        SH600004  SH600006  SH600017  SH600022  SH600026  SH600037  \
            date
            2016-01-05  0.001543  0.001570  0.002732  0.001320  0.003000       NaN
            2016-01-06  0.001538  0.001569  0.002770  0.001417  0.002945       NaN
            2016-01-07  0.001555  0.001546  0.002772  0.001393  0.002904       NaN
            2016-01-08  0.001564  0.001527  0.002791  0.001506  0.002948       NaN
            2016-01-11  0.001597  0.001476  0.002738  0.001493  0.003043       NaN

    stock_group_df : pd.DataFrame
        描述股票分组的数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 单元格值为分组ID
        - 示例(行业分组):
            instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008  \
            datetime
            2016-01-05  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
            2016-01-06  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
            2016-01-07  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
            2016-01-08  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
            2016-01-11  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0

    stock_ret_df : pd.DataFrame
        描述股票收益的数据框
        - 每行对应一个交易日
        - 每列对应一只股票
        - 单元格值为股票收益
        - 示例:
            instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008  \
            datetime
            2016-01-05  0.007795  0.022070  0.099099  0.024707  0.009473  0.016216
            2016-01-06 -0.032597 -0.075205 -0.098361 -0.098985 -0.099707 -0.098936
            2016-01-07 -0.001142  0.022544  0.100000  0.004225  0.000651  0.047226
            2016-01-08 -0.025157 -0.047244 -0.038567 -0.098177 -0.099609 -0.074408
            2016-01-11  0.023460  0.004959 -0.034384  0.018663  0.014461  0.010962

    返回值
    ----------
    tuple
        包含两个数据框的元组:
        1. group_weight_df: 分组权重数据框
        2. group_ret_df: 分组收益数据框

    实现逻辑
    ----------
    1. 首先调用decompose_portofolio_weight获取分组权重和组内股票权重
    2. 计算每个分组的收益
    3. 处理无权重分配时的收益为NaN的情况
    """
    all_group = np.unique(stock_group_df.values.flatten())
    all_group = all_group[~np.isnan(all_group)]

    group_weight, stock_weight_in_group = decompose_portofolio_weight(stock_weight_df, stock_group_df)

    group_ret = {}
    for group_key, val in stock_weight_in_group.items():
        stock_weight_in_group_start_date = min(val.index)
        stock_weight_in_group_end_date = max(val.index)

        temp_stock_ret_df = stock_ret_df[
            (stock_ret_df.index >= stock_weight_in_group_start_date)
            & (stock_ret_df.index <= stock_weight_in_group_end_date)
        ]

        group_ret[group_key] = (temp_stock_ret_df * val).sum(axis=1)
        # If no weight is assigned, then the return of group will be np.nan
        group_ret[group_key][group_weight[group_key] == 0.0] = np.nan

    group_weight_df = pd.DataFrame(group_weight)
    group_ret_df = pd.DataFrame(group_ret)
    return group_weight_df, group_ret_df


def get_daily_bin_group(bench_values, stock_values, group_n):
    """
    根据基准股票值进行每日分组

    参数
    ----------
    bench_values : pd.Series
        基准股票值序列
        - 索引为股票代码
    stock_values : pd.Series
        投资组合股票值序列
        - 索引为股票代码
    group_n : int
        分组数量

    返回值
    ----------
    pd.Series
        分组结果序列
        - 与stock_values相同大小和索引
        - 值为分组ID(1到group_n)
        - 第1组包含最大值

    实现逻辑
    ----------
    1. 根据基准股票值计算分位数分割点
    2. 将投资组合股票分配到对应分组
    3. 处理边界值为无穷大
    """
    stock_group = stock_values.copy()

    # get the bin split points based on the daily proportion of benchmark
    split_points = np.percentile(bench_values[~bench_values.isna()], np.linspace(0, 100, group_n + 1))
    # Modify the biggest uppper bound and smallest lowerbound
    split_points[0], split_points[-1] = -np.inf, np.inf
    for i, (lb, up) in enumerate(zip(split_points, split_points[1:])):
        stock_group.loc[stock_values[(stock_values >= lb) & (stock_values < up)].index] = group_n - i
    return stock_group


def get_stock_group(stock_group_field_df, bench_stock_weight_df, group_method, group_n=None):
    if group_method == "category":
        # use the value of the benchmark as the category
        return stock_group_field_df
    elif group_method == "bins":
        assert group_n is not None
        # place the values into `group_n` fields.
        # Each bin corresponds to a category.
        new_stock_group_df = stock_group_field_df.copy().loc[
            bench_stock_weight_df.index.min() : bench_stock_weight_df.index.max()
        ]
        for idx, row in (~bench_stock_weight_df.isna()).iterrows():
            bench_values = stock_group_field_df.loc[idx, row[row].index]
            new_stock_group_df.loc[idx] = get_daily_bin_group(
                bench_values,
                stock_group_field_df.loc[idx],
                group_n=group_n,
            )
        return new_stock_group_df


def brinson_pa(
    positions,
    bench="SH000905",
    group_field="industry",
    group_method="category",
    group_n=None,
    deal_price="vwap",
    freq="day",
):
    """Brinson收益归因分析

    参数
    ----------
    positions : dict
        回测类产生的持仓字典
    bench : str, optional
        用于比较的基准指数，默认为'SH000905'
        TODO: 如果未设置基准，则使用等权重
    group_field : str, optional
        用于资产分组的字段，默认为'industry'
                        `industry`和`market_value`是常用字段
    group_method : str, optional
        分组方法，可选'category'或'bins'，默认为'category'
        用于设置资产分配的分组方法
         `bin`方法会将值分成`group_n`个区间，每个区间代表一个分组
    group_n : int, optional
        分组数量，仅在group_method=='bins'时使用

    返回值
    ----------
    pd.DataFrame
        包含三列的数据框：
        - RAA(资产配置超额收益)
        - RSS(个股选择超额收益)
        - RTotal(总超额收益)
        每行对应一个交易日，值对应下一个交易日的收益
        Brinson收益归因分析的中间信息
    """
    # group_method will decide how to group the group_field.
    dates = sorted(positions.keys())

    start_date, end_date = min(dates), max(dates)

    bench_stock_weight = get_benchmark_weight(bench, start_date, end_date, freq)

    # The attributes for allocation will not
    if not group_field.startswith("$"):
        group_field = "$" + group_field
    if not deal_price.startswith("$"):
        deal_price = "$" + deal_price

    # 注意：当前版本中，某些停牌股票的属性(如市值)为NAN，因此需要获取更多日期数据向前填充NAN
    shift_start_date = start_date - datetime.timedelta(days=250)
    instruments = D.list_instruments(
        D.instruments(market="all"),
        start_time=shift_start_date,
        end_time=end_date,
        as_list=True,
        freq=freq,
    )
    stock_df = D.features(
        instruments,
        [group_field, deal_price],
        start_time=shift_start_date,
        end_time=end_date,
        freq=freq,
    )
    stock_df.columns = [group_field, "deal_price"]

    stock_group_field = stock_df[group_field].unstack().T
    # 注意：某些停牌股票的属性为NAN
    stock_group_field = stock_group_field.fillna(method="ffill")
    stock_group_field = stock_group_field.loc[start_date:end_date]

    stock_group = get_stock_group(stock_group_field, bench_stock_weight, group_method, group_n)

    deal_price_df = stock_df["deal_price"].unstack().T
    deal_price_df = deal_price_df.fillna(method="ffill")

    # 注意：
    # 这里的收益率与报告中的收益率会略有不同
    # 此处持仓是在交易日结束时以收盘价调整的
    stock_ret = (deal_price_df - deal_price_df.shift(1)) / deal_price_df.shift(1)
    stock_ret = stock_ret.shift(-1).loc[start_date:end_date]

    port_stock_weight_df = get_stock_weight_df(positions)

    # decomposing the portofolio
    port_group_weight_df, port_group_ret_df = decompose_portofolio(port_stock_weight_df, stock_group, stock_ret)
    bench_group_weight_df, bench_group_ret_df = decompose_portofolio(bench_stock_weight, stock_group, stock_ret)

    # if the group return of the portofolio is NaN, replace it with the market
    # value
    mod_port_group_ret_df = port_group_ret_df.copy()
    mod_port_group_ret_df[mod_port_group_ret_df.isna()] = bench_group_ret_df

    Q1 = (bench_group_weight_df * bench_group_ret_df).sum(axis=1)
    Q2 = (port_group_weight_df * bench_group_ret_df).sum(axis=1)
    Q3 = (bench_group_weight_df * mod_port_group_ret_df).sum(axis=1)
    Q4 = (port_group_weight_df * mod_port_group_ret_df).sum(axis=1)

    return (
        pd.DataFrame(
            {
                "RAA": Q2 - Q1,  # The excess profit from the assets allocation
                "RSS": Q3 - Q1,  # The excess profit from the stocks selection
                # The excess profit from the interaction of assets allocation and stocks selection
                "RIN": Q4 - Q3 - Q2 + Q1,
                "RTotal": Q4 - Q1,  # The totoal excess profit
            },
        ),
        {
            "port_group_ret": port_group_ret_df,
            "port_group_weight": port_group_weight_df,
            "bench_group_ret": bench_group_ret_df,
            "bench_group_weight": bench_group_weight_df,
            "stock_group": stock_group,
            "bench_stock_weight": bench_stock_weight,
            "port_stock_weight": port_stock_weight_df,
            "stock_ret": stock_ret,
        },
    )
