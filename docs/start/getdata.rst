.. _getdata:

==============
数据获取
==============

.. currentmodule:: qlib

简介
============

用户可以使用 ``Qlib`` 获取股票数据。以下示例演示了基本的用户接口。

示例
========

``QLib`` 初始化：

.. note:: 为了获取数据，用户需要先用 `qlib.init` 初始化 ``Qlib``。请参考 `初始化 <initialization.html>`_。

如果用户已按照 `初始化 <initialization.html>`_ 的步骤下载了数据，应使用以下代码初始化 qlib：

.. code-block:: python

    >> import qlib
    >> qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')


加载指定时间范围和频率的交易日历：

.. code-block:: python

   >> from qlib.data import D
   >> D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2]
   [Timestamp('2010-01-04 00:00:00'), Timestamp('2010-01-05 00:00:00')]

将给定市场名称解析为股票池配置：

.. code-block:: python

   >> from qlib.data import D
   >> D.instruments(market='all')
   {'market': 'all', 'filter_pipe': []}

加载指定时间范围内某股票池的成分股：

.. code-block:: python

   >> from qlib.data import D
   >> instruments = D.instruments(market='csi300')
   >> D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6]
   ['SH600036', 'SH600110', 'SH600087', 'SH600900', 'SH600089', 'SZ000912']

根据名称过滤器从基础市场动态加载成分股：

.. code-block:: python

   >> from qlib.data import D
   >> from qlib.data.filter import NameDFilter
   >> nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
   >> instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter])
   >> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)
   ['SH600655', 'SH601555']

根据表达式过滤器从基础市场动态加载成分股：

.. code-block:: python

   >> from qlib.data import D
   >> from qlib.data.filter import ExpressionDFilter
   >> expressionDFilter = ExpressionDFilter(rule_expression='$close>2000')
   >> instruments = D.instruments(market='csi300', filter_pipe=[expressionDFilter])
   >> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)
   ['SZ000651', 'SZ000002', 'SH600655', 'SH600570']

更多关于过滤器的细节，请参考 `过滤器 API <../component/data.html>`_。

加载指定时间范围内某些股票的特征：

.. code-block:: python

   >> from qlib.data import D
   >> instruments = ['SH600000']
   >> fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
   >> D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head().to_string()
   '                           $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low
   ... instrument  datetime
   ... SH600000    2010-01-04  86.778313  16162960.0       88.825928        88.061483    2.907631
   ...             2010-01-05  87.433578  28117442.0       86.778313        87.679273    3.235252
   ...             2010-01-06  85.713585  23632884.0       87.433578        86.641825    1.720009
   ...             2010-01-07  83.788803  20813402.0       85.713585        85.645322    3.030487
   ...             2010-01-08  84.730675  16044853.0       83.788803        84.744354    2.047623'

加载指定时间范围内某股票池的特征：

.. note:: 启用缓存后，qlib 数据服务器会一直为请求的股票池和字段缓存数据，首次请求可能比未启用缓存时更慢。但之后相同股票池和字段的请求会命中缓存，即使请求的时间区间变化也会更快。

.. code-block:: python

   >> from qlib.data import D
   >> from qlib.data.filter import NameDFilter, ExpressionDFilter
   >> nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
   >> expressionDFilter = ExpressionDFilter(rule_expression='$close>Ref($close,1)')
   >> instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter, expressionDFilter])
   >> fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
   >> D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head().to_string()
   '                              $close        $volume  Ref($close, 1)  Mean($close, 3)  $high-$low
   ... instrument  datetime
   ... SH600655    2010-01-04  2699.567383  158193.328125     2619.070312      2626.097738  124.580566
   ...             2010-01-08  2612.359619   77501.406250     2584.567627      2623.220133   83.373047
   ...             2010-01-11  2712.982422  160852.390625     2612.359619      2636.636556  146.621582
   ...             2010-01-12  2788.688232  164587.937500     2712.982422      2704.676758  128.413818
   ...             2010-01-13  2790.604004  145460.453125     2788.688232      2764.091553  128.413818'

更多关于特征的细节，请参考 `特征 API <../component/data.html>`_。

.. note:: 在客户端调用 `D.features()` 时，使用参数 `disk_cache=0` 可跳过数据集缓存，使用 `disk_cache=1` 可生成并使用数据集缓存。此外，在服务器端调用时，用户可用 `disk_cache=2` 更新数据集缓存。


当你构建复杂表达式时，将所有表达式写在一个字符串里可能不太方便。
例如，下面的表达式看起来很长很复杂：

.. code-block:: python

   >> from qlib.data import D
   >> data = D.features(["sh600519"], ["(($high / $close) + ($open / $close)) * (($high / $close) + ($open / $close)) / (($high / $close) + ($open / $close))"], start_time="20200101")


但字符串并不是实现表达式的唯一方式。你也可以用代码实现表达式。
下面是一个与上例等价的代码实现：

.. code-block:: python

   >> from qlib.data.ops import *
   >> f1 = Feature("high") / Feature("close")
   >> f2 = Feature("open") / Feature("close")
   >> f3 = f1 + f2
   >> f4 = f3 * f3 / f3

   >> data = D.features(["sh600519"], [f4], start_time="20200101")
   >> data.head()


API 参考
===
如需了解更多关于数据的使用方法，请参见 API 参考：`数据 API <../reference/api.html#data>`_
