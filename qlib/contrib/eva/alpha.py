"""
这里是一批评估函数。

未来应仔细重新设计接口。
"""

import pandas as pd
from typing import Tuple
from qlib import get_module_logger
from qlib.utils.paral import complex_parallel, DelayedDict
from joblib import Parallel, delayed


def calc_long_short_prec(
    pred: pd.Series, label: pd.Series, date_col="datetime", quantile: float = 0.2, dropna=False, is_alpha=False
) -> Tuple[pd.Series, pd.Series]:
    """
    计算多空操作的准确率


    :param pred/label: 索引为**pd.MultiIndex**，索引名称为**[datetime, instruments]**；列名称为**[score]**。

            .. code-block:: python
                                                  score
                datetime            instrument
                2020-12-01 09:30:00 SH600068    0.553634
                                    SH600195    0.550017
                                    SH600276    0.540321
                                    SH600584    0.517297
                                    SH600715    0.544674
    label :
        标签
    date_col :
        日期列

    返回
    -------
    (pd.Series, pd.Series)
        时间维度上的多头准确率和空头准确率
    """
    if is_alpha:
        label = label - label.groupby(level=date_col, group_keys=False).mean()
    if int(1 / quantile) >= len(label.index.get_level_values(1).unique()):
        raise ValueError("Need more instruments to calculate precision")

    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)

    group = df.groupby(level=date_col, group_keys=False)

    def N(x):
        return int(len(x) * quantile)

    # 找出预测值的最高/最低分位数，并将其作为多空目标
    long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label)
    short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label)

    groupll = long.groupby(date_col, group_keys=False)
    l_dom = groupll.apply(lambda x: x > 0)
    l_c = groupll.count()

    groups = short.groupby(date_col, group_keys=False)
    s_dom = groups.apply(lambda x: x < 0)
    s_c = groups.count()
    return (l_dom.groupby(date_col, group_keys=False).sum() / l_c), (
        s_dom.groupby(date_col, group_keys=False).sum() / s_c
    )


def calc_long_short_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算多空收益

    注意：
        `label`必须是原始股票收益。

    参数
    ----------
    pred : pd.Series
        股票预测值
    label : pd.Series
        股票收益
    date_col : str
        日期时间索引名称
    quantile : float
        多空分位数

    返回
    ----------
    long_short_r : pd.Series
        每日多空收益
    long_avg_r : pd.Series
        每日多头平均收益
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col, group_keys=False)

    def N(x):
        return int(len(x) * quantile)

    r_long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label.mean())
    r_short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label.mean())
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg


def pred_autocorr(pred: pd.Series, lag=1, inst_col="instrument", date_col="datetime"):
    """预测自相关

    限制：
    - 如果日期时间不是连续密集的，相关性将基于相邻日期计算。(有些用户可能期望NaN)

    :param pred: pd.Series，格式如下
                instrument  datetime
                SH600000    2016-01-04   -0.000403
                            2016-01-05   -0.000753
                            2016-01-06   -0.021801
                            2016-01-07   -0.065230
                            2016-01-08   -0.062465
    :类型 pred: pd.Series
    :param lag: 滞后阶数
    """
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
        get_module_logger("pred_autocorr").warning(f"Only the first column in {pred.columns} of `pred` is kept")
    pred_ustk = pred.sort_index().unstack(inst_col)
    corr_s = {}
    for (idx, cur), (_, prev) in zip(pred_ustk.iterrows(), pred_ustk.shift(lag).iterrows()):
        corr_s[idx] = cur.corr(prev)
    corr_s = pd.Series(corr_s).sort_index()
    return corr_s


def pred_autocorr_all(pred_dict, n_jobs=-1, **kwargs):
    """
    计算pred_dict的自相关

    参数
    ----------
    pred_dict : dict
        类似{<方法名称>: <预测值>}的字典
    kwargs :
        所有这些参数将传递给pred_autocorr
    """
    ac_dict = {}
    for k, pred in pred_dict.items():
        ac_dict[k] = delayed(pred_autocorr)(pred, **kwargs)
    return complex_parallel(Parallel(n_jobs=n_jobs, verbose=10), ac_dict)


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> (pd.Series, pd.Series):
    """计算IC值

    参数
    ----------
    pred :
        预测值
    label :
        标签
    date_col :
        日期列

    返回
    -------
    (pd.Series, pd.Series)
        IC值和排序IC值
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"]))
    ric = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


def calc_all_ic(pred_dict_all, label, date_col="datetime", dropna=False, n_jobs=-1):
    """
    计算所有IC值

    参数
    ----------
    pred_dict_all :
        类似{<方法名称>: <预测值>}的字典
    label:
        标签值的pd.Series

    返回
    -------
    {'Q2+IND_z': {'ic': <IC序列类似>
                          2016-01-04   -0.057407
                          ...
                          2020-05-28    0.183470
                          2020-05-29    0.171393
                  'ric': <排序IC序列类似>
                          2016-01-04   -0.040888
                          ...
                          2020-05-28    0.236665
                          2020-05-29    0.183886
                  }
    ...}
    """
    pred_all_ics = {}
    for k, pred in pred_dict_all.items():
        pred_all_ics[k] = DelayedDict(["ic", "ric"], delayed(calc_ic)(pred, label, date_col=date_col, dropna=dropna))
    pred_all_ics = complex_parallel(Parallel(n_jobs=n_jobs, verbose=10), pred_all_ics)
    return pred_all_ics
