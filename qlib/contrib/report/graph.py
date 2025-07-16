# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import importlib
from typing import Iterable

import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot


class BaseGraph:
    _name = None

    def __init__(
        self, df: pd.DataFrame = None, layout: dict = None, graph_kwargs: dict = None, name_dict: dict = None, **kwargs
    ):
        """

        :param df: 数据DataFrame
        :param layout: 图表布局配置
        :param graph_kwargs: 图表参数
        :param name_dict: 名称映射字典
        :param kwargs:
            layout: dict
                go.Layout的参数
            graph_kwargs: dict
                图表参数，例如：go.Bar(**graph_kwargs)
        """
        self._df = df

        self._layout = dict() if layout is None else layout
        self._graph_kwargs = dict() if graph_kwargs is None else graph_kwargs
        self._name_dict = name_dict

        self.data = None

        self._init_parameters(**kwargs)
        self._init_data()

    def _init_data(self):
        """

        :return:
        """
        if self._df.empty:
            raise ValueError("df is empty.")

        self.data = self._get_data()

    def _init_parameters(self, **kwargs):
        """

        :param kwargs
        """

        # Instantiate graphics parameters
        self._graph_type = self._name.lower().capitalize()

        # Displayed column name
        if self._name_dict is None:
            self._name_dict = {_item: _item for _item in self._df.columns}

    @staticmethod
    def get_instance_with_graph_parameters(graph_type: str = None, **kwargs):
        """

        :param graph_type:
        :param kwargs:
        :return:
        """
        try:
            _graph_module = importlib.import_module("plotly.graph_objs")
            _graph_class = getattr(_graph_module, graph_type)
        except AttributeError:
            _graph_module = importlib.import_module("qlib.contrib.report.graph")
            _graph_class = getattr(_graph_module, graph_type)
        return _graph_class(**kwargs)

    @staticmethod
    def show_graph_in_notebook(figure_list: Iterable[go.Figure] = None):
        """

        :param figure_list:
        :return:
        """
        py.init_notebook_mode()
        for _fig in figure_list:
            # NOTE: displays figures: https://plotly.com/python/renderers/
            # default: plotly_mimetype+notebook
            # support renderers: import plotly.io as pio; print(pio.renderers)
            renderer = None
            try:
                # in notebook
                _ipykernel = str(type(get_ipython()))
                if "google.colab" in _ipykernel:
                    renderer = "colab"
            except NameError:
                pass

            _fig.show(renderer=renderer)

    def _get_layout(self) -> go.Layout:
        """

        :return:
        """
        return go.Layout(**self._layout)

    def _get_data(self) -> list:
        """

        :return:
        """

        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type, x=self._df.index, y=self._df[_col], name=_name, **self._graph_kwargs
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data

    @property
    def figure(self) -> go.Figure:
        """

        :return:
        """
        _figure = go.Figure(data=self.data, layout=self._get_layout())
        # NOTE: Use the default theme from plotly version 3.x, template=None
        _figure["layout"].update(template=None)
        return _figure


class ScatterGraph(BaseGraph):
    _name = "scatter"


class BarGraph(BaseGraph):
    _name = "bar"


class DistplotGraph(BaseGraph):
    _name = "distplot"

    def _get_data(self):
        """

        :return:
        """
        _t_df = self._df.dropna()
        _data_list = [_t_df[_col] for _col in self._name_dict]
        _label_list = list(self._name_dict.values())
        _fig = create_distplot(_data_list, _label_list, show_rug=False, **self._graph_kwargs)

        return _fig["data"]


class HeatmapGraph(BaseGraph):
    _name = "heatmap"

    def _get_data(self):
        """

        :return:
        """
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type,
                x=self._df.columns,
                y=self._df.index,
                z=self._df.values.tolist(),
                **self._graph_kwargs,
            )
        ]
        return _data


class HistogramGraph(BaseGraph):
    _name = "histogram"

    def _get_data(self):
        """

        :return:
        """
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type, x=self._df[_col], name=_name, **self._graph_kwargs
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data


class SubplotsGraph:
    """创建与 df.plot(subplots=True) 相同的子图

    `plotly.tools.subplots` 的简单封装
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        kind_map: dict = None,
        layout: dict = None,
        sub_graph_layout: dict = None,
        sub_graph_data: list = None,
        subplots_kwargs: dict = None,
        **kwargs,
    ):
        """

        :param df: 数据表格（pd.DataFrame）

        :param kind_map: 字典，子图类型及参数
            例如: dict(kind='ScatterGraph', kwargs=dict())

        :param layout: `go.Layout` 的参数

        :param sub_graph_layout: 每个子图的布局，与 'layout' 类似

        :param sub_graph_data: 每个子图的实例化参数
            例如: [(column_name, instance_parameters), ]

            column_name: 字符串或 go.Figure 对象

            实例化参数:

                - row: 整数，子图所在的行

                - col: 整数，子图所在的列

                - name: 字符串，显示名称，默认为 df 中的列名

                - kind: 字符串，图表类型，默认为 `kind` 参数，例如: bar, scatter, ...

                - graph_kwargs: 字典，图表参数，默认为 {}，用于 `go.Bar(**graph_kwargs)`

        :param subplots_kwargs: `plotly.tools.make_subplots` 的原始参数

                - shared_xaxes: 布尔值，是否共享 x 轴，默认为 False

                - shared_yaxes: 布尔值，是否共享 y 轴，默认为 False

                - vertical_spacing: 浮点数，垂直间距，默认为 0.3 / 行数

                - subplot_titles: 列表，子图标题，默认为 []
                    若 `sub_graph_data` 为 None，将根据 `df.columns` 生成标题，此参数将被忽略


                - specs: 列表，详见 `make_subplots` 文档

                - rows: 整数，子图网格的行数，默认为 1
                    若 `sub_graph_data` 为 None，将根据 df 生成行数，此参数将被忽略

                - cols: 整数，子图网格的列数，默认为 1
                    若 `sub_graph_data` 为 None，将根据 df 生成列数，此参数将被忽略


        :param kwargs: 其他参数

        """

        self._df = df
        self._layout = layout
        self._sub_graph_layout = sub_graph_layout

        self._kind_map = kind_map
        if self._kind_map is None:
            self._kind_map = dict(kind="ScatterGraph", kwargs=dict())

        self._subplots_kwargs = subplots_kwargs
        if self._subplots_kwargs is None:
            self._init_subplots_kwargs()

        self.__cols = self._subplots_kwargs.get("cols", 2)  # pylint: disable=W0238
        self.__rows = self._subplots_kwargs.get(  # pylint: disable=W0238
            "rows", math.ceil(len(self._df.columns) / self.__cols)
        )

        self._sub_graph_data = sub_graph_data
        if self._sub_graph_data is None:
            self._init_sub_graph_data()

        self._init_figure()

    def _init_sub_graph_data(self):
        """
        初始化子图数据

        :return: None
        """
        self._sub_graph_data = []
        self._subplot_titles = []

        for i, column_name in enumerate(self._df.columns):
            row = math.ceil((i + 1) / self.__cols)
            _temp = (i + 1) % self.__cols
            col = _temp if _temp else self.__cols
            res_name = column_name.replace("_", " ")
            _temp_row_data = (
                column_name,
                dict(
                    row=row,
                    col=col,
                    name=res_name,
                    kind=self._kind_map["kind"],
                    graph_kwargs=self._kind_map["kwargs"],
                ),
            )
            self._sub_graph_data.append(_temp_row_data)
            self._subplot_titles.append(res_name)

    def _init_subplots_kwargs(self):
        """
        初始化子图参数

        :return: None
        """
        # 默认列数、行数
        _cols = 2
        _rows = math.ceil(len(self._df.columns) / 2)
        self._subplots_kwargs = dict()
        self._subplots_kwargs["rows"] = _rows
        self._subplots_kwargs["cols"] = _cols
        self._subplots_kwargs["shared_xaxes"] = False
        self._subplots_kwargs["shared_yaxes"] = False
        self._subplots_kwargs["vertical_spacing"] = 0.3 / _rows
        self._subplots_kwargs["print_grid"] = False
        self._subplots_kwargs["subplot_titles"] = self._df.columns.tolist()

    def _init_figure(self):
        """
        初始化图表对象

        :return: None
        """
        self._figure = make_subplots(**self._subplots_kwargs)

        for column_name, column_map in self._sub_graph_data:
            if isinstance(column_name, go.Figure):
                _graph_obj = column_name
            elif isinstance(column_name, str):
                temp_name = column_map.get("name", column_name.replace("_", " "))
                kind = column_map.get("kind", self._kind_map.get("kind", "ScatterGraph"))
                _graph_kwargs = column_map.get("graph_kwargs", self._kind_map.get("kwargs", {}))
                _graph_obj = BaseGraph.get_instance_with_graph_parameters(
                    kind,
                    **dict(
                        df=self._df.loc[:, [column_name]],
                        name_dict={column_name: temp_name},
                        graph_kwargs=_graph_kwargs,
                    ),
                )
            else:
                raise TypeError()

            row = column_map["row"]
            col = column_map["col"]

            _graph_data = getattr(_graph_obj, "data")
            # for _item in _graph_data:
            #     _item.pop('xaxis', None)
            #     _item.pop('yaxis', None)

            for _g_obj in _graph_data:
                self._figure.add_trace(_g_obj, row=row, col=col)

        if self._sub_graph_layout is not None:
            for k, v in self._sub_graph_layout.items():
                self._figure["layout"][k].update(v)

        # 注意: 使用 plotly 3.x 版本的默认主题: template=None
        self._figure["layout"].update(template=None)
        self._figure["layout"].update(self._layout)

    @property
    def figure(self):
        return self._figure
