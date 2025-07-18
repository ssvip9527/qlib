# 版权所有 (c) Microsoft Corporation。
# 基于 MIT 许可证授权。

import pandas as pd

from ..graph import SubplotsGraph, BaseGraph


def _calculate_maximum(df: pd.DataFrame, is_ex: bool = False):
    """
    计算最大回撤的起始和结束日期

    :param df: 数据框
    :param is_ex: 是否计算超额收益
    :return: 起始日期和结束日期
    """
    if is_ex:
        end_date = df["cum_ex_return_wo_cost_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_ex_return_wo_cost"].idxmax()
    else:
        end_date = df["return_wo_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_return_wo_cost"].idxmax()
    return start_date, end_date


def _calculate_mdd(series):
    """
    计算最大回撤

    :param series: 序列数据
    :return: 最大回撤序列
    """
    return series - series.cummax()


def _calculate_report_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算报告所需数据

    :param df: 输入数据框
    :return: 包含累积收益和回撤的报告数据框
    """
    index_names = df.index.names
    df.index = df.index.strftime("%Y-%m-%d")

    report_df = pd.DataFrame()

    report_df["cum_bench"] = df["bench"].cumsum()
    report_df["cum_return_wo_cost"] = df["return"].cumsum()
    report_df["cum_return_w_cost"] = (df["return"] - df["cost"]).cumsum()
    # report_df['cum_return'] - report_df['cum_return'].cummax()
    report_df["return_wo_mdd"] = _calculate_mdd(report_df["cum_return_wo_cost"])
    report_df["return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"]).cumsum())

    report_df["cum_ex_return_wo_cost"] = (df["return"] - df["bench"]).cumsum()
    report_df["cum_ex_return_w_cost"] = (df["return"] - df["bench"] - df["cost"]).cumsum()
    report_df["cum_ex_return_wo_cost_mdd"] = _calculate_mdd((df["return"] - df["bench"]).cumsum())
    report_df["cum_ex_return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"] - df["bench"]).cumsum())
    # return_wo_mdd , return_w_cost_mdd,  cum_ex_return_wo_cost_mdd, cum_ex_return_w

    report_df["turnover"] = df["turnover"]
    report_df.sort_index(ascending=True, inplace=True)

    report_df.index.names = index_names
    return report_df


def _report_figure(df: pd.DataFrame) -> [list, tuple]:
    """

    :param df:
    :return:
    """

    # Get data
    report_df = _calculate_report_data(df)

    # Maximum Drawdown
    max_start_date, max_end_date = _calculate_maximum(report_df)
    ex_max_start_date, ex_max_end_date = _calculate_maximum(report_df, True)

    index_name = report_df.index.name
    _temp_df = report_df.reset_index()
    _temp_df.loc[-1] = 0
    _temp_df = _temp_df.shift(1)
    _temp_df.loc[0, index_name] = "T0"
    _temp_df.set_index(index_name, inplace=True)
    _temp_df.iloc[0] = 0
    report_df = _temp_df

    # Create figure
    _default_kind_map = dict(kind="ScatterGraph", kwargs={"mode": "lines+markers"})
    _temp_fill_args = {"fill": "tozeroy", "mode": "lines+markers"}
    _column_row_col_dict = [
        ("cum_bench", dict(row=1, col=1)),
        ("cum_return_wo_cost", dict(row=1, col=1)),
        ("cum_return_w_cost", dict(row=1, col=1)),
        ("return_wo_mdd", dict(row=2, col=1, graph_kwargs=_temp_fill_args)),
        ("return_w_cost_mdd", dict(row=3, col=1, graph_kwargs=_temp_fill_args)),
        ("cum_ex_return_wo_cost", dict(row=4, col=1)),
        ("cum_ex_return_w_cost", dict(row=4, col=1)),
        ("turnover", dict(row=5, col=1)),
        ("cum_ex_return_w_cost_mdd", dict(row=6, col=1, graph_kwargs=_temp_fill_args)),
        ("cum_ex_return_wo_cost_mdd", dict(row=7, col=1, graph_kwargs=_temp_fill_args)),
    ]

    _subplot_layout = dict()
    for i in range(1, 8):
        # yaxis
        _subplot_layout.update({"yaxis{}".format(i): dict(zeroline=True, showline=True, showticklabels=True)})
        _show_line = i == 7
        _subplot_layout.update({"xaxis{}".format(i): dict(showline=_show_line, type="category", tickangle=45)})

    _layout_style = dict(
        height=1200,
        title=" ",
        shapes=[
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": max_start_date,
                "y0": 0.55,
                "x1": max_end_date,
                "y1": 1,
                "fillcolor": "#d3d3d3",
                "opacity": 0.3,
                "line": {
                    "width": 0,
                },
            },
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": ex_max_start_date,
                "y0": 0,
                "x1": ex_max_end_date,
                "y1": 0.55,
                "fillcolor": "#d3d3d3",
                "opacity": 0.3,
                "line": {
                    "width": 0,
                },
            },
        ],
    )

    _subplot_kwargs = dict(
        shared_xaxes=True,
        vertical_spacing=0.01,
        rows=7,
        cols=1,
        row_width=[1, 1, 1, 3, 1, 1, 3],
        print_grid=False,
    )
    figure = SubplotsGraph(
        df=report_df,
        layout=_layout_style,
        sub_graph_data=_column_row_col_dict,
        subplots_kwargs=_subplot_kwargs,
        kind_map=_default_kind_map,
        sub_graph_layout=_subplot_layout,
    ).figure
    return (figure,)


def report_graph(report_df: pd.DataFrame, show_notebook: bool = True) -> [list, tuple]:
    """display backtest report

        Example:


            .. code-block:: python

                import qlib
                import pandas as pd
                from qlib.utils.time import Freq
                from qlib.utils import flatten_dict
                from qlib.backtest import backtest, executor
                from qlib.contrib.evaluate import risk_analysis
                from qlib.contrib.strategy import TopkDropoutStrategy

                # init qlib
                qlib.init(provider_uri=<qlib data dir>)

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

                # strategy object
                strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
                # executor object
                executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
                # backtest
                portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
                analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
                # backtest info
                report_normal_df, positions_normal = portfolio_metric_dict.get(analysis_freq)

                qcr.analysis_position.report_graph(report_normal_df)

    :param report_df: **df.index.name** must be **date**, **df.columns** must contain **return**, **turnover**, **cost**, **bench**.


            .. code-block:: python

                            return      cost        bench       turnover
                date
                2017-01-04  0.003421    0.000864    0.011693    0.576325
                2017-01-05  0.000508    0.000447    0.000721    0.227882
                2017-01-06  -0.003321   0.000212    -0.004322   0.102765
                2017-01-09  0.006753    0.000212    0.006874    0.105864
                2017-01-10  -0.000416   0.000440    -0.003350   0.208396


    :param show_notebook: whether to display graphics in notebook, the default is **True**.
    :return: if show_notebook is True, display in notebook; else return **plotly.graph_objs.Figure** list.
    """
    report_df = report_df.copy()
    fig_list = _report_figure(report_df)
    if show_notebook:
        BaseGraph.show_graph_in_notebook(fig_list)
    else:
        return fig_list
