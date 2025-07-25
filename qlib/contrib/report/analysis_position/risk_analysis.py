# 版权所有 (c) Microsoft Corporation。
# 基于 MIT 许可证授权。

from typing import Iterable

import pandas as pd

import plotly.graph_objs as py

from ...evaluate import risk_analysis

from ..graph import SubplotsGraph, ScatterGraph


def _get_risk_analysis_data_with_report(
    report_normal_df: pd.DataFrame,
    # report_long_short_df: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """
    获取带有报告的风险分析数据

    :param report_normal_df: 报告数据框
    :param report_long_short_df: 报告数据框（已注释）
    :param date: 日期字符串
    :return: 风险分析数据框
    """

    analysis = dict()
    # if not report_long_short_df.empty:
    #     analysis["pred_long"] = risk_analysis(report_long_short_df["long"])
    #     analysis["pred_short"] = risk_analysis(report_long_short_df["short"])
    #     analysis["pred_long_short"] = risk_analysis(report_long_short_df["long_short"])

    if not report_normal_df.empty:
        analysis["excess_return_without_cost"] = risk_analysis(report_normal_df["return"] - report_normal_df["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]
        )
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    analysis_df["date"] = date
    return analysis_df


def _get_all_risk_analysis(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    将风险数据框转换为标准格式

    :param risk_df: 风险数据框
    :return: 标准格式的风险分析数据框
    """
    if risk_df is None:
        return pd.DataFrame()
    risk_df = risk_df.unstack()
    risk_df.columns = risk_df.columns.droplevel(0)
    return risk_df.drop("mean", axis=1)


def _get_monthly_risk_analysis_with_report(report_normal_df: pd.DataFrame) -> pd.DataFrame:
    """Get monthly analysis data

    :param report_normal_df:
    # :param report_long_short_df:
    :return:
    """

    # Group by month
    report_normal_gp = report_normal_df.groupby(
        [report_normal_df.index.year, report_normal_df.index.month], group_keys=False
    )
    # report_long_short_gp = report_long_short_df.groupby(
    #     [report_long_short_df.index.year, report_long_short_df.index.month], group_keys=False
    # )

    gp_month = sorted(set(report_normal_gp.size().index))

    _monthly_df = pd.DataFrame()
    for gp_m in gp_month:
        _m_report_normal = report_normal_gp.get_group(gp_m)
        # _m_report_long_short = report_long_short_gp.get_group(gp_m)

        if len(_m_report_normal) < 3:
            # The month's data is less than 3, not displayed
            # FIXME: If the trading day of a month is less than 3 days, a breakpoint will appear in the graph
            continue
        month_days = pd.Timestamp(year=gp_m[0], month=gp_m[1], day=1).days_in_month
        _temp_df = _get_risk_analysis_data_with_report(
            _m_report_normal,
            # _m_report_long_short,
            pd.Timestamp(year=gp_m[0], month=gp_m[1], day=month_days),
        )
        _monthly_df = pd.concat([_monthly_df, _temp_df], sort=False)

    return _monthly_df


def _get_monthly_analysis_with_feature(monthly_df: pd.DataFrame, feature: str = "annualized_return") -> pd.DataFrame:
    """

    :param monthly_df:
    :param feature:
    :return:
    """
    _monthly_df_gp = monthly_df.reset_index().groupby(["level_1"], group_keys=False)

    _name_df = _monthly_df_gp.get_group(feature).set_index(["level_0", "level_1"])
    _temp_df = _name_df.pivot_table(index="date", values=["risk"], columns=_name_df.index)
    _temp_df.columns = map(lambda x: "_".join(x[-1]), _temp_df.columns)
    _temp_df.index = _temp_df.index.strftime("%Y-%m")

    return _temp_df


def _get_risk_analysis_figure(analysis_df: pd.DataFrame) -> Iterable[py.Figure]:
    """Get analysis graph figure

    :param analysis_df:
    :return:
    """
    if analysis_df is None:
        return []

    _figure = SubplotsGraph(
        _get_all_risk_analysis(analysis_df),
        kind_map=dict(kind="BarGraph", kwargs={}),
        subplots_kwargs={"rows": 1, "cols": 4},
    ).figure
    return (_figure,)


def _get_monthly_risk_analysis_figure(report_normal_df: pd.DataFrame) -> Iterable[py.Figure]:
    """Get analysis monthly graph figure

    :param report_normal_df:
    :param report_long_short_df:
    :return:
    """

    # if report_normal_df is None and report_long_short_df is None:
    #     return []
    if report_normal_df is None:
        return []

    # if report_normal_df is None:
    #     report_normal_df = pd.DataFrame(index=report_long_short_df.index)

    # if report_long_short_df is None:
    #     report_long_short_df = pd.DataFrame(index=report_normal_df.index)

    _monthly_df = _get_monthly_risk_analysis_with_report(
        report_normal_df=report_normal_df,
        # report_long_short_df=report_long_short_df,
    )

    for _feature in ["annualized_return", "max_drawdown", "information_ratio", "std"]:
        _temp_df = _get_monthly_analysis_with_feature(_monthly_df, _feature)
        yield ScatterGraph(
            _temp_df,
            layout=dict(title=_feature, xaxis=dict(type="category", tickangle=45)),
            graph_kwargs={"mode": "lines+markers"},
        ).figure


def risk_analysis_graph(
    analysis_df: pd.DataFrame = None,
    report_normal_df: pd.DataFrame = None,
    report_long_short_df: pd.DataFrame = None,
    show_notebook: bool = True,
) -> Iterable[py.Figure]:
    """生成分析图表和月度分析

        示例:


            .. code-block:: python

                import qlib
                import pandas as pd
                from qlib.utils.time import Freq
                from qlib.utils import flatten_dict
                from qlib.backtest import backtest, executor
                from qlib.contrib.evaluate import risk_analysis
                from qlib.contrib.strategy import TopkDropoutStrategy

                # 初始化qlib
                qlib.init(provider_uri=<qlib数据目录>)

                CSI300_BENCH = "SH000300"
                FREQ = "day"
                STRATEGY_CONFIG = {
                    "topk": 50,
                    "n_drop": 5,
                    # pred_score, pd.Series
                    "signal": pred_score,
                }

                EXECUTOR_CONFIG = {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                }

                backtest_config = {
                    "start_time": "2017-01-01",
                    "end_time": "2020-08-01",
                    "account": 100000000,
                    "benchmark": CSI300_BENCH,
                    "exchange_kwargs": {
                        "freq": FREQ,
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                }

                # 策略对象
                strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
                # 执行器对象
                executor_obj = executor.SimulatorExecutor(** EXECUTOR_CONFIG)
                # 回测
                portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
                analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
                # 回测信息
                report_normal_df, positions_normal = portfolio_metric_dict.get(analysis_freq)
                analysis = dict()
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_normal_df["return"] - report_normal_df["bench"], freq=analysis_freq
                )
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"], freq=analysis_freq
                )

                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                analysis_position.risk_analysis_graph(analysis_df, report_normal_df)



    :param analysis_df: 分析数据，索引为**pd.MultiIndex**；列名为**[risk]**.


            .. code-block:: python

                                                                  risk
                excess_return_without_cost mean               0.000692
                                           std                0.005374
                                           annualized_return  0.174495
                                           information_ratio  2.045576
                                           max_drawdown      -0.079103
                excess_return_with_cost    mean               0.000499
                                           std                0.005372
                                           annualized_return  0.125625
                                           information_ratio  1.473152
                                           max_drawdown      -0.088263


    :param report_normal_df: **df.index.name**必须为**date**，df.columns必须包含**return**、**turnover**、**cost**、**bench**.


            .. code-block:: python

                            return      cost        bench       turnover
                date
                2017-01-04  0.003421    0.000864    0.011693    0.576325
                2017-01-05  0.000508    0.000447    0.000721    0.227882
                2017-01-06  -0.003321   0.000212    -0.004322   0.102765
                2017-01-09  0.006753    0.000212    0.006874    0.105864
                2017-01-10  -0.000416   0.000440    -0.003350   0.208396


    :param report_long_short_df: **df.index.name**必须为**date**，df.columns包含**long**、**short**、**long_short**.


            .. code-block:: python

                            long        short       long_short
                date
                2017-01-04  -0.001360   0.001394    0.000034
                2017-01-05  0.002456    0.000058    0.002514
                2017-01-06  0.000120    0.002739    0.002859
                2017-01-09  0.001436    0.001838    0.003273
                2017-01-10  0.000824    -0.001944   -0.001120


    :param show_notebook: 是否在notebook中显示图形，默认为**True**.
        若为True，在notebook中显示图形
        若为False，返回图形对象
    :return: 图形对象列表
    """
    _figure_list = list(_get_risk_analysis_figure(analysis_df)) + list(
        _get_monthly_risk_analysis_figure(
            report_normal_df,
            # report_long_short_df,
        )
    )
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(_figure_list)
    else:
        return _figure_list
