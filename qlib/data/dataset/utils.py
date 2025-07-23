# 版权所有 (c) Microsoft Corporation。
# 基于 MIT 许可证授权。
from __future__ import annotations
import pandas as pd
from typing import Union, List, TYPE_CHECKING
from qlib.utils import init_instance_by_config

if TYPE_CHECKING:
    from qlib.data.dataset import DataHandler


def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    """

    获取给定`level`在`df`中的级别索引

    参数
    ----------
    df : pd.DataFrame
        数据
    level : Union[str, int]
        索引级别

    返回
    -------
    int:
        多级索引中的级别索引
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # 注意：如果数据中未指定级别索引，默认级别索引将为('datetime', 'instrument')
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"不支持这种输入类型")


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list, pd.Index],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    """
    使用`selector`和`level`从`data`中获取数据

        假设selector已被正确处理。
        `fetch_df_by_index`仅负责获取正确的级别

    参数
    ----------
    selector : Union[pd.Timestamp, slice, str, list, pd.Index]
        选择器
    level : Union[int, str]
        应用选择器的级别

    返回
    -------
    pd.DataFrame:
        给定索引的数据。
    """
    # level = None -> 直接使用选择器
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]
    # 尝试获取正确的索引
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[pd.IndexSlice[idx_slc],]  # noqa: E231
        else:  # pylint: disable=W0120
            return df
    else:
        return df.loc[pd.IndexSlice[idx_slc],]  # noqa: E231


def fetch_df_by_col(df: pd.DataFrame, col_set: Union[str, List[str]]) -> pd.DataFrame:
    from .handler import DataHandler  # pylint: disable=C0415

    if not isinstance(df.columns, pd.MultiIndex) or col_set == DataHandler.CS_RAW:
        return df
    elif col_set == DataHandler.CS_ALL:
        return df.droplevel(axis=1, level=0)
    else:
        return df.loc(axis=1)[col_set]


def convert_index_format(df: Union[pd.DataFrame, pd.Series], level: str = "datetime") -> Union[pd.DataFrame, pd.Series]:
    """
    根据以下规则转换df.MultiIndex的格式：
        - 如果`level`是df.MultiIndex的第一级，则不执行任何操作
        - 如果`level`是df.MultiIndex的第二级，则交换索引级别。

    注意：
        df.MultiIndex的级别数应为2

    参数
    ----------
    df : Union[pd.DataFrame, pd.Series]
        原始DataFrame/Series
    level : str, 可选
        将被转换为第一级的级别，默认为"datetime"

    返回
    -------
    Union[pd.DataFrame, pd.Series]
        转换后的DataFrame/Series
    """

    if get_level_index(df, level=level) == 1:
        df = df.swaplevel().sort_index()
    return df


def init_task_handler(task: dict) -> DataHandler:
    """
    **就地**初始化任务的处理器部分

    参数
    ----------
    task : dict
        要处理的任务

    返回
    -------
    Union[DataHandler, None]:
        返回初始化后的处理器实例
    """
    # avoid recursive import
    from .handler import DataHandler  # pylint: disable=C0415

    h_conf = task["dataset"]["kwargs"].get("handler")
    if h_conf is not None:
        handler = init_instance_by_config(h_conf, accept_types=DataHandler)
        task["dataset"]["kwargs"]["handler"] = handler
        return handler
    else:
        raise ValueError("The task does not contains a handler part.")
