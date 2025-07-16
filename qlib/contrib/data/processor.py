import numpy as np

from ...log import TimeInspector
from ...data.dataset.processor import Processor, get_group_columns


class ConfigSectionProcessor(Processor):
    """
    此处理器专为Alpha158设计，未来将被简单处理器替代。
    """

    def __init__(self, fields_group=None, **kwargs):
        super().__init__()
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

        self.fields_group = None

    def __call__(self, df):
        return self._transform(df)

    def _transform(self, df):
        def _label_norm(x):
            x = x - x.mean()  # 复制
            x /= x.std()
            if self.clip_label_outlier:
                x.clip(-3, 3, inplace=True)
            if self.fillna_label:
                x.fillna(0, inplace=True)
            return x

        def _feature_norm(x):
            x = x - x.median()  # 复制
            x /= x.abs().median() * 1.4826
            if self.clip_feature_outlier:
                x.clip(-3, 3, inplace=True)
            if self.shrink_feature_outlier:
                x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
                x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
            if self.fillna_feature:
                x.fillna(0, inplace=True)
            return x

        TimeInspector.set_time_mark()

        # 复制关注部分并将其改为单层级
        selected_cols = get_group_columns(df, self.fields_group)
        df_focus = df[selected_cols].copy()
        if len(df_focus.columns.levels) > 1:
            df_focus = df_focus.droplevel(level=0)

        # 标签
        cols = df_focus.columns[df_focus.columns.str.contains("^LABEL")]
        df_focus[cols] = df_focus[cols].groupby(level="datetime", group_keys=False).apply(_label_norm)

        # 特征
        cols = df_focus.columns[df_focus.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df_focus[cols] = (
            df_focus[cols].apply(lambda x: x**0.25).groupby(level="datetime", group_keys=False).apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^KLOW2|^KUP2")]
        df_focus[cols] = (
            df_focus[cols].apply(lambda x: x**0.5).groupby(level="datetime", group_keys=False).apply(_feature_norm)
        )

        _cols = [
            "KMID",
            "KSFT",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VWAP",
            "ROC",
            "MA",
            "BETA",
            "RESI",
            "QTLU",
            "QTLD",
            "RSV",
            "SUMP",
            "SUMN",
            "SUMD",
            "VSUMP",
            "VSUMN",
            "VSUMD",
        ]
        pat = "|".join(["^" + x for x in _cols])
        cols = df_focus.columns[df_focus.columns.str.contains(pat) & (~df_focus.columns.isin(["HIGH0", "LOW0"]))]
        df_focus[cols] = df_focus[cols].groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
        df_focus[cols] = df_focus[cols].apply(np.log).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^RSQR")]
        df_focus[cols] = df_focus[cols].fillna(0).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^MAX|^HIGH0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (x - 1) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^MIN|^LOW0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (1 - x) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^CORR|^CORD")]
        df_focus[cols] = df_focus[cols].apply(np.exp).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^WVMA")]
        df_focus[cols] = df_focus[cols].apply(np.log1p).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        df[selected_cols] = df_focus.values

        TimeInspector.log_cost_time("数据预处理完成。")

        return df
