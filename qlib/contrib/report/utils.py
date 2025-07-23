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
    此函数`猜测`移除日期时间索引中间隔所需的范围断点。
    它基本上计算`连续`日期时间索引与给定索引之间的差异。

    有关plotly中`rangebreaks`参数的更多详细信息，请参见
    https://plotly.com/python/reference/layout/xaxis/#layout-xaxis-rangebreaks

    参数
    ----------
    dt_index: pd.DatetimeIndex
    数据的日期时间。

    返回值
    -------
    要传递给plotly轴的`rangebreaks`。

    """
    dt_idx = dt_index.sort_values()
    gaps = dt_idx[1:] - dt_idx[:-1]
    min_gap = gaps.min()
    gaps_to_break = {}
    for gap, d in zip(gaps, dt_idx[:-1]):
        if gap > min_gap:
            gaps_to_break.setdefault(gap - min_gap, []).append(d + min_gap)
    return [dict(values=v, dvalue=int(k.total_seconds() * 1000)) for k, v in gaps_to_break.items()]
