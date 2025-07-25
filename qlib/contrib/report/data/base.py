# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
此模块负责数据分析

假设条件
- 每个特征单独分析

"""
import pandas as pd
from qlib.log import TimeInspector
from qlib.contrib.report.utils import sub_fig_generator


class FeaAnalyser:
    def __init__(self, dataset: pd.DataFrame):
        """

        Parameters
        ----------
        dataset : pd.DataFrame

            We often have multiple columns for dataset. Each column corresponds to one sub figure.
            There will be a datatime column in the index levels.
            Aggretation will be used for more summarized metrics overtime.
            Here is an example of data:

            .. code-block::

                                            return
                datetime   instrument
                2007-02-06 equity_tpx     0.010087
                           equity_spx     0.000786
        """
        self._dataset = dataset
        with TimeInspector.logt("calc_stat_values"):
            self.calc_stat_values()

    def calc_stat_values(self):
        pass

    def plot_single(self, col, ax):
        raise NotImplementedError(f"This type of input is not supported")

    def skip(self, col):
        return False

    def plot_all(self, *args, **kwargs):
        ax_gen = iter(sub_fig_generator(*args, **kwargs))
        for col in self._dataset:
            if not self.skip(col):
                ax = next(ax_gen)
                self.plot_single(col, ax)
