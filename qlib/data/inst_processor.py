import abc
import json
import pandas as pd


class InstProcessor:
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        """
        处理数据

        注意：**处理器可能会就地修改`df`的内容！！！**
        用户应在外部保留数据的副本

        参数
        ----------
        df : pd.DataFrame
            处理器的原始数据框或前一个处理器的结果。
        """

    def __str__(self):
        return f"{self.__class__.__name__}:{json.dumps(self.__dict__, sort_keys=True, default=str)}"
