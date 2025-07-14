.. _data:

==================================
数据层：数据框架与使用方法
==================================

简介
============

``数据层``提供了用户友好的API来管理和检索数据。它提供了高性能的数据基础设施。

数据层专为量化投资设计。例如，用户可以使用``数据层``轻松构建公式化alpha因子。更多详情请参考`构建公式化Alpha因子 <../advanced/alpha.html>`_。

``数据层``的介绍包括以下部分：

- 数据准备
- 数据API
- 数据加载器
- 数据处理器
- 数据集
- 缓存
- 数据和缓存文件结构

以下是Qlib数据工作流的典型示例：

- 用户下载数据并将其转换为Qlib格式（文件后缀为`.bin`）。在此步骤中，通常仅将一些基本数据（如OHLCV）存储在磁盘上。
- 基于Qlib的表达式引擎创建一些基本特征（例如"Ref($close, 60) / $close"，表示过去60个交易日的收益率）。表达式引擎支持的运算符可在`此处 <https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py>`__找到。此步骤通常在Qlib的`数据加载器 <https://qlib.readthedocs.io/en/latest/component/data.html#data-loader>`_中实现，它是`数据处理器 <https://qlib.readthedocs.io/en/latest/component/data.html#data-handler>`_的一个组件。
- 如果用户需要更复杂的数据处理（例如数据归一化），`数据处理器 <https://qlib.readthedocs.io/en/latest/component/data.html#data-handler>`_支持用户自定义处理器来处理数据（一些预定义的处理器可在`此处 <https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py>`__找到）。这些处理器与表达式引擎中的运算符不同，它们专为一些难以用表达式引擎运算符支持的复杂数据处理方法而设计。
- 最后，`数据集 <https://qlib.readthedocs.io/en/latest/component/data.html#dataset>`_负责从数据处理器处理后的数据中准备模型特定的数据集

数据准备
============

Qlib格式数据
----------------

我们专门设计了一种数据结构来管理金融数据，详细信息请参考Qlib论文中的`文件存储设计部分 <https://arxiv.org/abs/2009.11189>`_。
此类数据将以`.bin`为文件后缀存储（我们称之为`.bin`文件、`.bin`格式或Qlib格式）。`.bin`文件专为金融数据的科学计算而设计。

``Qlib``提供了两种现成的数据集，可通过此`链接 <https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py>`__访问：

========================  =================  ================
数据集                    美国市场           中国市场
========================  =================  ================
Alpha360                  √                  √

Alpha158                  √                  √
========================  =================  ================

此外，``Qlib``还提供了高频数据集。用户可以通过此`链接 <https://github.com/microsoft/qlib/tree/main/examples/highfreq>`__运行高频数据集示例。

Qlib格式数据集
-------------------
``Qlib``提供了`.bin`格式的现成数据集，用户可以使用脚本``scripts/get_data.py``下载中国股票数据集，如下所示。用户也可以使用numpy加载`.bin`文件来验证数据。
价量数据与实际交易价格不同，因为它们经过了**复权处理**（`复权价格 <https://www.investopedia.com/terms/a/adjusted_closing_price.asp>`_）。您可能会发现不同数据源的复权价格有所不同，这是因为不同数据源的复权方式可能不同。Qlib在复权时将每只股票第一个交易日的价格归一化为1。
用户可以使用`$factor`获取原始交易价格（例如，`$close / $factor`可获取原始收盘价）。

以下是关于Qlib价格复权的一些讨论：

- https://github.com/microsoft/qlib/issues/991#issuecomment-1075252402


.. code-block:: bash

    # 下载日线数据
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

    # 下载1分钟数据
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1min --region cn --interval 1min

除中国股票数据外，``Qlib``还包括美国股票数据集，可通过以下命令下载：

.. code-block:: bash

    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us

运行上述命令后，用户可以在``~/.qlib/qlib_data/cn_data``目录和``~/.qlib/qlib_data/us_data``目录中分别找到Qlib格式的中国股票和美国股票数据。

``Qlib``还在``scripts/data_collector``中提供了脚本，帮助用户从互联网爬取最新数据并转换为Qlib格式。

当使用此数据集初始化``Qlib``后，用户可以使用它构建和评估自己的模型。更多详情请参考`初始化 <../start/initialization.html>`_。

日线数据自动更新
----------------------------------------

  **建议用户先手动更新一次数据（--trading_date 2021-05-25），然后再设置为自动更新。**

  更多信息请参考：`yahoo数据收集器 <https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#Automatic-update-of-daily-frequency-data>`_

  - 每个交易日自动更新数据到"qlib"目录（Linux）
      - 使用*crontab*：`crontab -e`
      - 设置定时任务：

        .. code-block:: bash

            * * * * 1-5 python <脚本路径> update_data_to_bin --qlib_data_1d_dir <用户数据目录>

        - **脚本路径**：*scripts/data_collector/yahoo/collector.py*

  - 手动更新数据

      .. code-block:: bash

        python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <用户数据目录> --trading_date <开始日期> --end_date <结束日期>

      - *trading_date*：交易日开始日期
      - *end_date*：交易日结束日期（不包含）



将CSV格式转换为Qlib格式
--------------------------------------

``Qlib``提供了脚本``scripts/dump_bin.py``，可以将**任何**CSV格式的数据转换为`.bin`文件（``Qlib``格式），只要数据格式正确。

除了下载准备好的演示数据外，用户还可以直接从数据收集器下载演示数据，如下所示，以参考CSV格式。
以下是一些示例：

日线数据：
  .. code-block:: bash

    python scripts/get_data.py download_data --file_name csv_data_cn.zip --target_dir ~/.qlib/csv_data/cn_data

1分钟数据：
  .. code-block:: bash

    python scripts/data_collector/yahoo/collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_1min --region CN --start 2021-05-20 --end 2021-05-23 --delay 0.1 --interval 1min --limit_nums 10

用户也可以提供自己的CSV格式数据。但是，CSV数据**必须满足**以下条件：

- CSV文件以特定股票命名，或者CSV文件包含股票名称列

    - 以股票名称命名CSV文件：`SH600000.csv`、`AAPL.csv`（不区分大小写）。

    - CSV文件包含股票名称列。用户在转换数据时**必须**指定列名。例如：

        .. code-block:: bash

            python scripts/dump_bin.py dump_all ... --symbol_field_name symbol

        其中数据格式如下：

            +-----------+-------+
            | symbol    | close |
            +===========+=======+
            | SH600000  | 120   |
            +-----------+-------+

- CSV文件**必须**包含日期列，并且在转换数据时，用户必须指定日期列名。例如：

    .. code-block:: bash

        python scripts/dump_bin.py dump_all ... --date_field_name date

    其中数据格式如下：

        +---------+------------+-------+------+----------+
        | symbol  | date       | close | open | volume   |
        +=========+============+=======+======+==========+
        | SH600000| 2020-11-01 | 120   | 121  | 12300000 |
        +---------+------------+-------+------+----------+
        | SH600000| 2020-11-02 | 123   | 120  | 12300000 |
        +---------+------------+-------+------+----------+


假设用户在目录``~/.qlib/csv_data/my_data``中准备了CSV格式的数据，可以运行以下命令开始转换：

.. code-block:: bash

    python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/my_data --qlib_dir ~/.qlib/qlib_data/my_data --include_fields open,close,high,low,volume,factor

有关将数据转换为`.bin`文件时支持的其他参数，用户可以通过运行以下命令查看：

.. code-block:: bash

    python dump_bin.py dump_all --help

转换后，用户可以在目录`~/.qlib/qlib_data/my_data`中找到Qlib格式的数据。

.. note::

    `--include_fields`参数应与CSV文件的列名相对应。``Qlib``提供的数据集列名至少应包括open、close、high、low、volume和factor。

    - `open`
        复权开盘价
    - `close`
        复权收盘价
    - `high`
        复权最高价
    - `low`
        复权最低价
    - `volume`
        复权成交量
    - `factor`
        复权因子。通常，``factor = 复权价格 / 原始价格``，复权价格参考：`拆股调整 <https://www.investopedia.com/terms/s/splitadjusted.asp>`_

    在``Qlib``数据处理的约定中，如果股票停牌，`open、close、high、low、volume、money和factor`将被设置为NaN。
    如果您想使用无法通过OHLCV计算的自定义alpha因子（如PE、EPS等），可以将其与OHLCV一起添加到CSV文件中，然后转换为Qlib格式数据。

数据健康检查
-------------------------------

``Qlib``提供了一个脚本来检查数据的健康状况。

- 主要检查点如下

    - 检查DataFrame中是否有任何数据缺失。

    - 检查OHLCV列中是否存在超过阈值的大幅阶跃变化。

    - 检查DataFrame中是否缺少任何必需列（OLHCV）。

    - 检查DataFrame中是否缺少'factor'列。

- 您可以运行以下命令检查数据是否健康。

    日线数据：
        .. code-block:: bash

            python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data

    1分钟数据：
        .. code-block:: bash

            python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data_1min --freq 1min

- 当然，您也可以添加一些参数来调整测试结果。

    - 可用参数如下：

        - freq：数据频率。

        - large_step_threshold_price：允许的最大价格变化

        - large_step_threshold_volume：允许的最大成交量变化。

        - missing_data_num：允许数据为空的最大值。

- 您可以运行以下命令检查数据是否健康。

    日线数据：
        .. code-block:: bash

            python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data --missing_data_num 30055 --large_step_threshold_volume 94485 --large_step_threshold_price 20

    1分钟数据：
        .. code-block:: bash

            python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data --freq 1min --missing_data_num 35806 --large_step_threshold_volume 3205452000000 --large_step_threshold_price 0.91

股票池（市场）
-------------------

``Qlib``将`股票池 <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml#L4>`_定义为股票列表及其日期范围。可以按以下方式导入预定义的股票池（如沪深300）。

.. code-block:: bash

    python collector.py --index_name CSI300 --qlib_dir <用户qlib数据目录> --method parse_instruments


多股票模式
--------------------

``Qlib``目前为用户提供两种不同的股票模式：中国股票模式和美国股票模式。以下是这两种模式的一些不同设置：

==============  =================  ================
地区            交易单位           涨跌幅限制
==============  =================  ================
中国            100                0.099

美国            1                  无
==============  =================  ================

`交易单位`定义了一次交易中可以使用的股票数量单位，`涨跌幅限制`定义了股票涨跌百分比的界限。

- 如果用户在A股模式下使用``Qlib``，需要A股数据。用户可以按照以下步骤在A股模式下使用``Qlib``：
    - 下载Qlib格式的A股数据，请参考章节`Qlib格式数据集 <#qlib-format-dataset>`_。
    - 初始化A股模式的``Qlib``
        假设用户在目录``~/.qlib/qlib_data/cn_data``中下载了Qlib格式数据。用户只需按以下方式初始化``Qlib``即可。

        .. code-block:: python

            from qlib.constant import REG_CN
            qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)


- 如果用户在美股模式下使用``Qlib``，需要美股数据。``Qlib``也提供了下载美股数据的脚本。用户可以按照以下步骤在美股模式下使用``Qlib``：
    - 下载Qlib格式的美股数据，请参考章节`Qlib格式数据集 <#qlib-format-dataset>`_。
    - 初始化美股模式的``Qlib``
        假设用户在目录``~/.qlib/qlib_data/us_data``中准备了Qlib格式数据。用户只需按以下方式初始化``Qlib``即可。

        .. code-block:: python

            from qlib.config import REG_US
            qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region=REG_US)


.. note::

    非常欢迎提交新数据源的PR！用户可以像`此处示例 <https://github.com/microsoft/qlib/tree/main/scripts>`_一样提交爬取数据的代码作为PR。然后我们将使用该代码在服务器上创建数据缓存，供其他用户直接使用。


数据API
========

数据检索
--------------
用户可以使用``qlib.data``中的API检索数据，请参考`数据检索 <../start/getdata.html>`_。

特征
-------

``Qlib``提供`Feature`（特征）和`ExpressionOps`（表达式操作）来根据用户需求获取特征。

- `Feature`（特征）
    从数据提供程序加载数据。用户可以获取诸如`$high`、`$low`、`$open`、`$close`等特征，这些特征应与`--include_fields`参数对应，请参考章节`将CSV格式转换为Qlib格式 <#converting-csv-format-into-qlib-format>`_。

- `ExpressionOps`（表达式操作）
    `ExpressionOps`将使用运算符进行特征构建。
    要了解更多关于``Operator``（运算符）的信息，请参考`运算符API <../reference/api.html#module-qlib.data.ops>`_。
    此外，``Qlib``支持用户定义自己的自定义``Operator``，示例已在``tests/test_register_ops.py``中给出。

要了解更多关于``Feature``的信息，请参考`特征API <../reference/api.html#module-qlib.data.base>`_。

过滤器
------
``Qlib``提供`NameDFilter`（名称动态过滤器）和`ExpressionDFilter`（表达式动态过滤器），用于根据用户需求过滤证券。

- `NameDFilter`（名称动态过滤器）
    名称动态证券过滤器。基于规定的名称格式过滤证券。需要名称规则正则表达式。

- `ExpressionDFilter`（表达式动态过滤器）
    表达式动态证券过滤器。基于特定表达式过滤证券。需要指示特定特征字段的表达式规则。

    - `基础特征过滤`：rule_expression = '$close/$open>5'
    - `横截面特征过滤`：rule_expression = '$rank($close)<10'
    - `时间序列特征过滤`：rule_expression = '$Ref($close, 3)>100'

以下是一个简单示例，展示如何在基本的``Qlib``工作流配置文件中使用过滤器：

.. code-block:: yaml

    filter: &filter
        filter_type: ExpressionDFilter
        rule_expression: "Ref($close, -2) / Ref($close, -1) > 1"
        filter_start_time: 2010-01-01
        filter_end_time: 2010-01-07
        keep: False

    data_handler_config: &data_handler_config
        start_time: 2010-01-01
        end_time: 2021-01-22
        fit_start_time: 2010-01-01
        fit_end_time: 2015-12-31
        instruments: *market
        filter_pipe: [*filter]

要了解更多关于``Filter``的信息，请参考`过滤器API <../reference/api.html#module-qlib.data.filter>`_。

参考
---------

要了解更多关于``数据API``的信息，请参考`数据API <../reference/api.html#data>`_。


数据加载器
===========

``Qlib``中的``Data Loader``（数据加载器）旨在从原始数据源加载原始数据。它将被加载并在``Data Handler``（数据处理器）模块中使用。

QlibDataLoader
--------------

``Qlib``中的``QlibDataLoader``类是一个允许用户从``Qlib``数据源加载原始数据的接口。

StaticDataLoader
----------------

``Qlib``中的``StaticDataLoader``类是一个允许用户从文件或提供的数据源加载原始数据的接口。


接口
---------

以下是``QlibDataLoader``类的一些接口：

.. autoclass:: qlib.data.dataset.loader.DataLoader
    :members: 
    :noindex:

API
---

要了解更多关于``数据加载器``的信息，请参考`数据加载器API <../reference/api.html#module-qlib.data.dataset.loader>`_。


数据处理器
============

``Qlib``中的``Data Handler``（数据处理器）模块旨在处理大多数模型将使用的常见数据处理方法。

用户可以通过``qrun``在自动工作流中使用``Data Handler``，更多详情请参考`工作流：工作流管理 <workflow.html>`_。

DataHandlerLP
-------------

除了在使用``qrun``的自动工作流中使用``Data Handler``外，``Data Handler``还可以作为独立模块使用，用户可以通过它轻松地预处理数据（标准化、去除NaN等）并构建数据集。

为此，``Qlib``提供了一个基类`qlib.data.dataset.DataHandlerLP <../reference/api.html#qlib.data.dataset.handler.DataHandlerLP>`_。该类的核心思想是：我们将拥有一些可学习的``Processors``（处理器），它们可以学习数据处理的参数（例如zscore归一化的参数）。当新数据到来时，这些“训练好的”``Processors``可以处理新数据，从而能够高效地处理实时数据。关于``Processors``的更多信息将在下一小节中列出。


接口
---------

以下是``DataHandlerLP``提供的一些重要接口：

.. autoclass:: qlib.data.dataset.handler.DataHandlerLP
    :members: __init__, fetch, get_cols
    :noindex:


如果用户希望通过配置加载特征和标签，可以定义一个新的处理器并调用``qlib.contrib.data.handler.Alpha158``的静态方法`parse_config_to_fields`。

此外，用户可以将``qlib.contrib.data.processor.ConfigSectionProcessor``传递给新处理器，该处理器为配置定义的特征提供一些预处理方法。


处理器
---------

``Qlib``中的``Processor``（处理器）模块设计为可学习的，负责处理数据处理任务，如`归一化`和`删除无/NaN特征/标签`。

``Qlib``提供以下``Processors``：

- ``DropnaProcessor``：删除N/A特征的`处理器`。
- ``DropnaLabel``：删除N/A标签的`处理器`。
- ``TanhProcess``：使用`tanh`处理噪声数据的`处理器`。
- ``ProcessInf``：处理无穷大值的`处理器`，将用列的平均值替换无穷大值。
- ``Fillna``：处理N/A值的`处理器`，将N/A值填充为0或其他给定数字。
- ``MinMaxNorm``：应用最小-最大归一化的`处理器`。
- ``ZscoreNorm``：应用z-score归一化的`处理器`。
- ``RobustZScoreNorm``：应用稳健z-score归一化的`处理器`。
- ``CSZScoreNorm``：应用横截面z-score归一化的`处理器`。
- ``CSRankNorm``：应用横截面排名归一化的`处理器`。
- ``CSZFillna``：以横截面方式通过列平均值填充N/A值的`处理器`。

用户也可以通过继承``Processor``的基类创建自己的`处理器`。有关所有处理器的实现详情，请参考（`处理器链接 <https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py>`_）。

要了解更多关于``Processor``的信息，请参考`处理器API <../reference/api.html#module-qlib.data.dataset.processor>`_。

示例
-------

``Data Handler``可以通过修改配置文件与``qrun``一起运行，也可以作为独立模块使用。

要了解更多关于如何使用``qrun``运行``Data Handler``的信息，请参考`工作流：工作流管理 <workflow.html>`_

Qlib提供了已实现的数据处理器`Alpha158`。以下示例展示如何将`Alpha158`作为独立模块运行。

.. note:: 用户需要先使用`qlib.init`初始化``Qlib``，请参考`初始化 <../start/initialization.html>`_。

.. code-block:: Python

    import qlib
    from qlib.contrib.data.handler import Alpha158

    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": "csi300",
    }

    if __name__ == "__main__":
        qlib.init()
        h = Alpha158(**data_handler_config)

        # 获取数据的所有列
        print(h.get_cols())

        # 获取所有标签
        print(h.fetch(col_set="label"))

        # 获取所有特征
        print(h.fetch(col_set="feature"))


.. note:: 在``Alpha158``中，``Qlib``使用标签`Ref($close, -2)/Ref($close, -1) - 1`，表示从T+1到T+2的变化，而不是`Ref($close, -1)/$close - 1`。原因是当获取中国股票的T日收盘价时，股票可以在T+1日买入并在T+2日卖出。

API
---

要了解更多关于``数据处理器``的信息，请参考`数据处理器API <../reference/api.html#module-qlib.data.dataset.handler>`_。


数据集
=======

``Qlib``中的``Dataset``（数据集）模块旨在为模型训练和推理准备数据。

该模块的设计动机是最大限度地提高不同模型处理适合自身数据的灵活性。此模块使模型能够以独特的方式处理其数据。例如，``GBDT``等模型可以很好地处理包含`nan`或`None`值的数据，而``MLP``等神经网络则会因此崩溃。

如果用户的模型需要以不同方式处理数据，可以实现自己的``Dataset``类。如果模型的数据处理没有特殊要求，可以直接使用``DatasetH``。

``DatasetH``类是带有`数据处理器`的`数据集`。以下是该类最重要的接口：

.. autoclass:: qlib.data.dataset.__init__.DatasetH
    :members:
    :noindex:

API
---

要了解更多关于``数据集``的信息，请参考`数据集API <../reference/api.html#dataset>`_。


缓存
=====

``Cache``（缓存）是一个可选模块，通过将一些常用数据保存为缓存文件来帮助加速数据提供。``Qlib``提供了一个`Memcache`类用于在内存中缓存最常用的数据，一个可继承的`ExpressionCache`类，以及一个可继承的`DatasetCache`类。

全局内存缓存
-------------------

`Memcache`是一种全局内存缓存机制，由三个`MemCacheUnit`实例组成，用于缓存**日历**、**证券**和**特征**。`MemCache`在`cache.py`中全局定义为`H`。用户可以使用`H['c'], H['i'], H['f']`来获取/设置`memcache`。

.. autoclass:: qlib.data.cache.MemCacheUnit
    :members:
    :noindex:

.. autoclass:: qlib.data.cache.MemCache
    :members:
    :noindex:


表达式缓存
---------------

`ExpressionCache`是一种缓存机制，用于保存诸如**Mean($close, 5)** 之类的表达式。用户可以通过以下步骤继承此基类来定义自己的表达式缓存机制：

- 重写`self._uri`方法以定义缓存文件路径的生成方式
- 重写`self._expression`方法以定义要缓存的数据以及如何缓存它。

以下是接口的详细信息：

.. autoclass:: qlib.data.cache.ExpressionCache
    :members:
    :noindex:

``Qlib``目前提供了已实现的磁盘缓存`DiskExpressionCache`，它继承自`ExpressionCache`。表达式数据将存储在磁盘中。

数据集缓存
------------

`DatasetCache`是一种用于保存数据集的缓存机制。特定数据集由股票池配置（或一系列证券，尽管不推荐）、表达式列表或静态特征字段、收集特征的开始时间和结束时间以及频率来规定。用户可以通过以下步骤继承此基类来定义自己的数据集缓存机制：

- 重写`self._uri`方法以定义缓存文件路径的生成方式
- 重写`self._expression`方法以定义要缓存的数据以及如何缓存它。

以下是接口的详细信息：

.. autoclass:: qlib.data.cache.DatasetCache
    :members:
    :noindex:

``Qlib``目前提供了已实现的磁盘缓存`DiskDatasetCache`，它继承自`DatasetCache`。数据集数据将存储在磁盘中。



数据和缓存文件结构
=============================

我们专门设计了一种文件结构来管理数据和缓存，详细信息请参考Qlib论文中的`文件存储设计部分 <https://arxiv.org/abs/2009.11189>`_。数据和缓存的文件结构如下所示。

.. code-block::

    - data/
        [原始数据] 由数据提供程序更新
        - calendars/
            - day.txt
        - instruments/
            - all.txt
            - csi500.txt
            - ...
        - features/
            - sh600000/
                - open.day.bin
                - close.day.bin
                - ...
            - ...
        [缓存数据] 当原始数据更新时更新
        - calculated features/
            - sh600000/
                - [hash(instrtument, field_expression, freq)]
                    - all-time expression -cache data file
                    - .meta : 记录证券名称、字段名称、频率和访问次数的元文件
            - ...
        - cache/
            - [hash(stockpool_config, field_expression_list, freq)]
                - all-time Dataset-cache data file
                - .meta : 记录股票池配置、字段名称和访问次数的元文件
                - .index : 记录所有日历行索引的索引文件
            - ...
