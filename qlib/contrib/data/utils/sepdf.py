# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pandas as pd
from typing import Dict, Iterable, Union


def align_index(df_dict, join):
    res = {}
    for k, df in df_dict.items():
        if join is not None and k != join:
            df = df.reindex(df_dict[join].index)
        res[k] = df
    return res


# 模拟pd.DataFrame类
class SepDataFrame:
    """
    （Sep）分离式DataFrame
    我们通常会拼接多个DataFrame一起处理（如特征、标签、权重、过滤器）。
    然而，它们最终通常会被单独使用。
    这会导致拼接和拆分数据时产生额外成本（在内存中重塑和复制数据非常昂贵）

    SepDataFrame尝试表现得像列带有MultiIndex的DataFrame
    """

    # TODO:
    # SepDataFrame尝试表现得像pandas DataFrame，但两者仍不相同
    # 欢迎贡献代码使其更加完善。

    def __init__(self, df_dict: Dict[str, pd.DataFrame], join: str, skip_align=False):
        """
        基于DataFrame字典初始化数据

        参数
        ----------
        df_dict : Dict[str, pd.DataFrame]
            DataFrame字典
        join : str
            数据连接方式
            将基于连接键重新索引DataFrame。
            如果join为None，则跳过重新索引步骤

        skip_align :
            在某些情况下，可通过跳过索引对齐来提高性能
        """
        self.join = join

        if skip_align:
            self._df_dict = df_dict
        else:
            self._df_dict = align_index(df_dict, join)

    @property
    def loc(self):
        return SDFLoc(self, join=self.join)

    @property
    def index(self):
        return self._df_dict[self.join].index

    def apply_each(self, method: str, skip_align=True, *args, **kwargs):
        """
        假设：
        - 原地修改方法会返回None
        """
        inplace = False
        df_dict = {}
        for k, df in self._df_dict.items():
            df_dict[k] = getattr(df, method)(*args, **kwargs)
            if df_dict[k] is None:
                inplace = True
        if not inplace:
            return SepDataFrame(df_dict=df_dict, join=self.join, skip_align=skip_align)

    def sort_index(self, *args, **kwargs):
        return self.apply_each("sort_index", True, *args, **kwargs)

    def copy(self, *args, **kwargs):
        return self.apply_each("copy", True, *args, **kwargs)

    def _update_join(self):
        if self.join not in self:
            if len(self._df_dict) > 0:
                self.join = next(iter(self._df_dict.keys()))
            else:
                # 注意：当所有键为空时，这将改变之前的重新索引行为
                self.join = None

    def __getitem__(self, item):
        # TODO: 处理MultiIndex时表现更接近pandas
        return self._df_dict[item]

    def __setitem__(self, item: str, df: Union[pd.DataFrame, pd.Series]):
        # TODO: 考虑连接行为
        if not isinstance(item, tuple):
            self._df_dict[item] = df
        else:
            # 注意：MultiIndex的边界情况
            _df_dict_key, *col_name = item
            col_name = tuple(col_name)
            if _df_dict_key in self._df_dict:
                if len(col_name) == 1:
                    col_name = col_name[0]
                self._df_dict[_df_dict_key][col_name] = df
            else:
                if isinstance(df, pd.Series):
                    if len(col_name) == 1:
                        col_name = col_name[0]
                    self._df_dict[_df_dict_key] = df.to_frame(col_name)
                else:
                    df_copy = df.copy()  # 避免修改df
                    df_copy.columns = pd.MultiIndex.from_tuples([(*col_name, *idx) for idx in df.columns.to_list()])
                    self._df_dict[_df_dict_key] = df_copy

    def __delitem__(self, item: str):
        del self._df_dict[item]
        self._update_join()

    def __contains__(self, item):
        return item in self._df_dict

    def __len__(self):
        return len(self._df_dict[self.join])

    def droplevel(self, *args, **kwargs):
        raise NotImplementedError(f"请实现`droplevel`方法")

    @property
    def columns(self):
        dfs = []
        for k, df in self._df_dict.items():
            df = df.head(0)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            dfs.append(df)
        return pd.concat(dfs, axis=1).columns

    # Useless methods
    @staticmethod
    def merge(df_dict: Dict[str, pd.DataFrame], join: str):
        all_df = df_dict[join]
        for k, df in df_dict.items():
            if k != join:
                all_df = all_df.join(df)
        return all_df


class SDFLoc:
    """模拟类"""

    def __init__(self, sdf: SepDataFrame, join):
        self._sdf = sdf
        self.axis = None
        self.join = join

    def __call__(self, axis):
        self.axis = axis
        return self

    def __getitem__(self, args):
        if self.axis == 1:
            if isinstance(args, str):
                return self._sdf[args]
            elif isinstance(args, (tuple, list)):
                new_df_dict = {k: self._sdf[k] for k in args}
                return SepDataFrame(new_df_dict, join=self.join if self.join in args else args[0], skip_align=True)
            else:
                raise NotImplementedError(f"不支持此类型的输入")
        elif self.axis == 0:
            return SepDataFrame(
                {k: df.loc(axis=0)[args] for k, df in self._sdf._df_dict.items()}, join=self.join, skip_align=True
            )
        else:
            df = self._sdf
            if isinstance(args, tuple):
                ax0, *ax1 = args
                if len(ax1) == 0:
                    ax1 = None
                if ax1 is not None:
                    df = df.loc(axis=1)[ax1]
                if ax0 is not None:
                    df = df.loc(axis=0)[ax0]
                return df
            else:
                return df.loc(axis=0)[args]


# Patch pandas DataFrame
# 欺骗isinstance使其接受SepDataFrame作为子类
import builtins


def _isinstance(instance, cls):
    if isinstance_orig(instance, SepDataFrame):  # pylint: disable=E0602  # noqa: F821
        if isinstance(cls, Iterable):
            for c in cls:
                if c is pd.DataFrame:
                    return True
        elif cls is pd.DataFrame:
            return True
    return isinstance_orig(instance, cls)  # pylint: disable=E0602  # noqa: F821


builtins.isinstance_orig = builtins.isinstance
builtins.isinstance = _isinstance

if __name__ == "__main__":
    sdf = SepDataFrame({}, join=None)
    print(isinstance(sdf, (pd.DataFrame,)))
    print(isinstance(sdf, pd.DataFrame))
