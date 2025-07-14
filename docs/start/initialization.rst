.. _initialization:

===================
Qlib Initialization
===================

.. currentmodule:: qlib


Initialization
==============

请按照以下步骤初始化``Qlib``。

下载并准备数据：执行以下命令下载股票数据。请注意，数据来源于`Yahoo Finance <https://finance.yahoo.com/lookup>`_，可能不是完美的。如果用户有高质量数据集，建议准备自己的数据。有关自定义数据集的更多信息，请参考`数据 <../component/data.html#converting-csv-format-into-qlib-format>`_。

    .. code-block:: bash

        python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

有关`get_data.py`的更多信息，请参考`数据准备 <../component/data.html#data-preparation>`_。


在调用其他API之前初始化Qlib：在Python中运行以下代码。

    .. code-block:: Python

        import qlib
        # region in [REG_CN, REG_US]
        from qlib.constant import REG_CN
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(provider_uri=provider_uri, region=REG_CN)

.. note::
   不要在Qlib的仓库目录中导入qlib包，否则可能会发生错误。

Parameters
-------------------

除了`provider_uri`和`region`之外，`qlib.init`还有其他参数。以下是`qlib.init`的几个重要参数（Qlib有很多配置，这里仅列出部分参数。更详细的设置可以在`此处 <https://github.com/microsoft/qlib/blob/main/qlib/config.py>`_找到）：

- `provider_uri`
    类型：字符串。Qlib数据的URI。例如，它可以是``get_data.py``加载的数据存储的位置。
- `region`
    类型：字符串，可选参数（默认：`qlib.constant.REG_CN`）。目前支持``qlib.constant.REG_US``（'us'）和``qlib.constant.REG_CN``（'cn'）。不同的`region`值将导致不同的股票市场模式。- ``qlib.constant.REG_US``：美国股票市场。- ``qlib.constant.REG_CN``：中国股票市场。不同模式将导致不同的交易限制和成本。区域只是`定义一批配置的快捷方式 <https://github.com/microsoft/qlib/blob/528f74af099bf6156e9480bcd2bb28e453231212/qlib/config.py#L249>`_，包括最小交易单位（``trade_unit``）、涨跌幅限制（``limit_threshold``）等。它不是必需的部分，如果现有区域设置不能满足要求，用户可以手动设置关键配置。
- `redis_host`
    类型：字符串，可选参数（默认："127.0.0.1"），`redis`的主机。锁和缓存机制依赖于redis。
- `redis_port`
    类型：整数，可选参数（默认：6379），`redis`的端口。

    .. note::

        The value of `region` should be aligned with the data stored in `provider_uri`. Currently, ``scripts/get_data.py`` only provides China stock market data. If users want to use the US stock market data, they should prepare their own US-stock data in `provider_uri` and switch to US-stock mode.

    .. note::

        如果Qlib无法通过`redis_host`和`redis_port`连接Redis，将不会使用缓存机制！有关详细信息，请参考`缓存 <../component/data.html#cache>`_。
- `exp_manager`
    类型：字典，可选参数，用于在qlib中使用的`实验管理器`设置。用户可以指定实验管理器类，以及所有实验的跟踪URI。但请注意，我们仅支持以下样式的字典作为`exp_manager`的输入。有关`exp_manager`的更多信息，用户可以参考`Recorder: Experiment Management <../component/recorder.html>`_。

    .. code-block:: Python

        # For example, if you want to set your tracking_uri to a <specific folder>, you can initialize qlib below
        qlib.init(provider_uri=provider_uri, region=REG_CN, exp_manager= {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "python_execution_path/mlruns",
                "default_exp_name": "Experiment",
            }
        })
- `mongo`
    类型：字典，可选参数，`MongoDB <https://www.mongodb.com/>`_ 的设置，将用于某些功能，如`任务管理 <../advanced/task_management.html>`_，具有高性能和集群处理能力。
    用户需要按照 `安装 <https://www.mongodb.com/try/download/community>`_ 中的步骤首先安装MongoDB，然后通过URI访问它。
    用户可以通过将 "task_url" 设置为类似 `"mongodb://%s:%s@%s" % (user, pwd, host + ":" + port)` 的字符串来使用凭据访问mongodb。

    .. code-block:: Python

        # For example, you can initialize qlib below
        qlib.init(provider_uri=provider_uri, region=REG_CN, mongo={
            "task_url": "mongodb://localhost:27017/",  # your mongo url
            "task_db_name": "rolling_db", # the database name of Task Management
        })

- `logging_level`
    系统的日志级别。

- `kernels`
    Qlib表达式引擎中计算特征时使用的进程数。当调试表达式计算异常时，将其设置为1非常有帮助。
