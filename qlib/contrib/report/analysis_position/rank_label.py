# 版权所有 (c) Microsoft Corporation。
# 基于 MIT 许可证授权。

import copy
from typing import Iterable

import pandas as pd
import plotly.graph_objs as go

from ..graph import ScatterGraph
from ..analysis_position.parse_position import get_position_data


def _get_figure_with_position(
    position: dict, label_data: pd.DataFrame, start_date=None, end_date=None
) -> Iterable[go.Figure]:
    """
    获取平均分析图表

    :param position: 持仓信息
    :param label_data: 标签数据
    :param start_date: 起始日期
    :param end_date: 结束日期
    :return: 平均分析图表的可迭代对象
    """
    _position_df = get_position_data(
        position,
        label_data,
        calculate_label_rank=True,
        start_date=start_date,
        end_date=end_date,
    )

    res_dict = dict()
    _pos_gp = _position_df.groupby(level=1, group_keys=False)
    for _item in _pos_gp:
        _date = _item[0]
        _day_df = _item[1]

        _day_value = res_dict.setdefault(_date, {})
        for _i, _name in {0: "Hold", 1: "Buy", -1: "Sell"}.items():
            _temp_df = _day_df[_day_df["status"] == _i]
            if _temp_df.empty:
                _day_value[_name] = 0
            else:
                _day_value[_name] = _temp_df["rank_label_mean"].values[0]

    _res_df = pd.DataFrame.from_dict(res_dict, orient="index")
    # FIXME: support HIGH-FREQ
    _res_df.index = _res_df.index.strftime("%Y-%m-%d")
    for _col in _res_df.columns:
        yield ScatterGraph(
            _res_df.loc[:, [_col]],
            layout=dict(
                title=_col,
                xaxis=dict(type="category", tickangle=45),
                yaxis=dict(title="lable-rank-ratio: %"),
            ),
            graph_kwargs=dict(mode="lines+markers"),
        ).figure


def rank_label_graph(
    position: dict,
    label_data: pd.DataFrame,
    start_date=None,
    end_date=None,
    show_notebook=True,
) -> Iterable[go.Figure]:
    """交易日股票买入、卖出和持有的排名百分比。
    每日交易的平均排名比率（类似于 **sell_df['label'].rank(ascending=False) / len(sell_df)**）

        示例：


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.evaluate import backtest
                from qlib.contrib.strategy import TopkDropoutStrategy

                # 回测参数
                bparas = {}
                bparas['limit_threshold'] = 0.095
                bparas['account'] = 1000000000

                sparas = {}
                sparas['topk'] = 50
                sparas['n_drop'] = 230
                strategy = TopkDropoutStrategy(**sparas)

                _, positions = backtest(pred_df, strategy, **bparas)

                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'], pred_df_dates.min(), pred_df_dates.max())
                features_df.columns = ['label']

                qcr.analysis_position.rank_label_graph(positions, features_df, pred_df_dates.min(), pred_df_dates.max())


    :param position: 持仓数据；**qlib.backtest.backtest** 的结果。
    :param label_data: **D.features** 的结果；索引是 **pd.MultiIndex**，索引名称是 **[instrument, datetime]**；列名是 **[label]**。

        **标签 T 是从 T 到 T+1 的变化**，建议使用 ``close``，例如：`D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'])`。


            .. code-block:: python

                                                label
                instrument  datetime
                SH600004        2017-12-11  -0.013502
                                2017-12-12  -0.072367
                                2017-12-13  -0.068605
                                2017-12-14  0.012440
                                2017-12-15  -0.102778


    :param start_date: 开始日期
    :param end_date: 结束日期
    :param show_notebook: **True** 或 **False**。如果为 True，则在 notebook 中显示图表，否则返回图表。
    :return: 图表对象
    """
    position = copy.deepcopy(position)
    label_data.columns = ["label"]
    _figures = _get_figure_with_position(position, label_data, start_date, end_date)
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(_figures)
    else:
        return _figures
