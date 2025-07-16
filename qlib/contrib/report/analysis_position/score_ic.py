# 版权所有 (c) Microsoft Corporation。
# 基于 MIT 许可证授权。

import pandas as pd

from ..graph import ScatterGraph
from ..utils import guess_plotly_rangebreaks


def _get_score_ic(pred_label: pd.DataFrame):
    """
    获取分数的IC值

    :param pred_label: 包含预测分数和标签的数据框
    :return: 包含IC和Rank IC的数据框
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level="datetime", group_keys=False).apply(lambda x: x["label"].corr(x["score"]))
    _rank_ic = concat_data.groupby(level="datetime", group_keys=False).apply(
        lambda x: x["label"].corr(x["score"], method="spearman")
    )
    return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})


def score_ic_graph(pred_label: pd.DataFrame, show_notebook: bool = True, **kwargs) -> [list, tuple]:
    """分数IC图表

        示例:


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.report import analysis_position
                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(D.instruments('csi500'), ['Ref($close, -2)/Ref($close, -1)-1'], pred_df_dates.min(), pred_df_dates.max())
                features_df.columns = ['label']
                pred_label = pd.concat([features_df, pred], axis=1, sort=True).reindex(features_df.index)
                analysis_position.score_ic_graph(pred_label)


    :param pred_label: 索引为**pd.MultiIndex**，索引名称为**[instrument, datetime]**；列名为**[score, label]**.


            .. code-block:: python

                instrument  datetime        score         label
                SH600004  2017-12-11     -0.013502       -0.013502
                            2017-12-12   -0.072367       -0.072367
                            2017-12-13   -0.068605       -0.068605
                            2017-12-14    0.012440        0.012440
                            2017-12-15   -0.102778       -0.102778


    :param show_notebook: whether to display graphics in notebook, the default is **True**.
    :return: if show_notebook is True, display in notebook; else return **plotly.graph_objs.Figure** list.
    """
    _ic_df = _get_score_ic(pred_label)

    _figure = ScatterGraph(
        _ic_df,
        layout=dict(
            title="Score IC",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(_ic_df.index))),
        ),
        graph_kwargs={"mode": "lines+markers"},
    ).figure
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([_figure])
    else:
        return (_figure,)
