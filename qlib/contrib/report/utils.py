# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import pandas as pd


def sub_fig_generator(sub_figsize=(3, 3), col_n=10, row_n=1, wspace=None, hspace=None, sharex=False, sharey=False):
    """sub_fig_generator.
    生成一个生成器，每行包含<col_n>个子图

    FIXME: 已知限制：
    - 最后一行不会自动绘制，请在函数外部处理

    参数
    ----------
    sub_figsize :
        <col_n> * <row_n>个子图中每个子图的尺寸
    col_n :
        每行的子图数量；生成<col_n>个子图后将创建新图形
    row_n :
        每列的子图数量
    wspace :
        每行子图之间的宽度间隔
    hspace :
        每列子图之间的高度间隔
        如果觉得拥挤，可以尝试设置为0.3

    返回值
    -------
    每次迭代返回形状为<col_n>的图形（已压缩）。
    """
    assert col_n > 1

    while True:
        fig, axes = plt.subplots(
            row_n, col_n, figsize=(sub_figsize[0] * col_n, sub_figsize[1] * row_n), sharex=sharex, sharey=sharey
        )
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        axes = axes.reshape(row_n, col_n)

        for col in range(col_n):
            res = axes[:, col].squeeze()
            if res.size == 1:
                res = res.item()
            yield res
        plt.show()


def guess_plotly_rangebreaks(dt_index: pd.DatetimeIndex):
    """
    This function `guesses` the rangebreaks required to remove gaps in datetime index.
    It basically calculates the difference between a `continuous` datetime index and index given.

    For more details on `rangebreaks` params in plotly, see
    https://plotly.com/python/reference/layout/xaxis/#layout-xaxis-rangebreaks

    Parameters
    ----------
    dt_index: pd.DatetimeIndex
    The datetimes of the data.

    Returns
    -------
    the `rangebreaks` to be passed into plotly axis.

    """
    dt_idx = dt_index.sort_values()
    gaps = dt_idx[1:] - dt_idx[:-1]
    min_gap = gaps.min()
    gaps_to_break = {}
    for gap, d in zip(gaps, dt_idx[:-1]):
        if gap > min_gap:
            gaps_to_break.setdefault(gap - min_gap, []).append(d + min_gap)
    return [dict(values=v, dvalue=int(k.total_seconds() * 1000)) for k, v in gaps_to_break.items()]
