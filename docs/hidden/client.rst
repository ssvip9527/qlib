.. _client:

Qlib 客户端-服务器框架
============================

.. currentmodule:: qlib

简介
------------
客户端-服务器框架旨在解决以下问题

- 集中管理数据。用户无需管理不同版本的数据。
- 减少需要生成的缓存量。
- 支持远程访问数据。

因此，我们设计了客户端-服务器框架来解决这些问题。
我们将维护一个服务器并提供数据。

您需要使用特定配置来初始化qlib以使用客户端-服务器框架。
以下是一个典型的初始化过程。

qlib ``init`` 常用参数；必须在客户端所在的服务器上安装 ``nfs-common``，执行：``sudo apt install nfs-common``：
    - ``provider_uri``: nfs服务器路径；格式为 ``host: data_dir``，例如：``172.23.233.89:/data2/gaochao/sync_qlib/qlib``。如果使用离线模式，可以是本地数据目录
    - ``mount_path``: 本地数据目录，``provider_uri`` 将被挂载到此目录
    - ``auto_mount``: 是否在qlib ``init`` 期间自动将 ``provider_uri`` 挂载到 ``mount_path``；您也可以手动挂载：sudo mount.nfs ``provider_uri`` ``mount_path``。如果在PAI上运行，建议设置 ``auto_mount=True``
    - ``flask_server``: 数据服务主机；如果您在内网，可以使用默认主机：172.23.233.89
    - ``flask_port``: 数据服务端口


如果在 10.150.144.153 或 10.150.144.154 服务器上运行，建议使用以下代码来 ``init`` qlib：

.. code-block:: python

   >>> import qlib
   >>> qlib.init(auto_mount=False, mount_path='/data/csdesign/qlib')
   >>> from qlib.data import D
   >>> D.features(['SH600000'], ['$close'], start_time='20080101', end_time='20090101').head()
    [39336:MainThread](2019-05-28 21:35:42,800) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:56] - provider_uri=172.23.233.89:/data2/gaochao/sync_qlib/qlib
    [39336:Thread-68](2019-05-28 21:35:42,809) INFO - Client - [client.py:28] - Connect to server ws://172.23.233.89:9710
    [39336:Thread-72](2019-05-28 21:35:43,489) INFO - Client - [client.py:31] - Disconnect from server!
    Opening /data/csdesign/qlib/cache/d239a3b191daa9a5b1b19a59beb47b33 in read-only mode
    Out[5]:
                               $close
    instrument datetime
    SH600000   2008-01-02  119.079704
               2008-01-03  113.120125
               2008-01-04  117.878860
               2008-01-07  124.505539
               2008-01-08  125.395004


如果在PAI上运行，建议使用以下代码来 ``init`` qlib：

.. code-block:: python

   >>> import qlib
   >>> qlib.init(auto_mount=True, mount_path='/data/csdesign/qlib', provider_uri='172.23.233.89:/data2/gaochao/sync_qlib/qlib')
   >>> from qlib.data import D
   >>> D.features(['SH600000'], ['$close'], start_time='20080101', end_time='20090101').head()
    [39336:MainThread](2019-05-28 21:35:42,800) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:56] - provider_uri=172.23.233.89:/data2/gaochao/sync_qlib/qlib
    [39336:Thread-68](2019-05-28 21:35:42,809) INFO - Client - [client.py:28] - Connect to server ws://172.23.233.89:9710
    [39336:Thread-72](2019-05-28 21:35:43,489) INFO - Client - [client.py:31] - Disconnect from server!
    Opening /data/csdesign/qlib/cache/d239a3b191daa9a5b1b19a59beb47b33 in read-only mode
    Out[5]:
                               $close
    instrument datetime
    SH600000   2008-01-02  119.079704
               2008-01-03  113.120125
               2008-01-04  117.878860
               2008-01-07  124.505539
               2008-01-08  125.395004


如果在Windows上运行，打开 **NFS** 功能并写入正确的 **mount_path**，建议使用以下代码来 ``init`` qlib：

1.Windows系统开启NFS功能
    * 打开 ``程序和功能``
    * 点击 ``启用或关闭Windows功能``
    * 向下滚动并勾选 ``NFS服务`` 选项，然后点击确定

    参考地址：https://graspingtech.com/mount-nfs-share-windows-10/
2.配置正确的mount_path
    * 在Windows中，挂载路径必须是不存在的路径且为根路径
        * 正确的路径格式例如：`H`、`i`...
        * 错误的路径格式例如：`C`、`C:/user/name`、`qlib_data`...

.. code-block:: python

   >>> import qlib
   >>> qlib.init(auto_mount=True, mount_path='H', provider_uri='172.23.233.89:/data2/gaochao/sync_qlib/qlib')
   >>> from qlib.data import D
   >>> D.features(['SH600000'], ['$close'], start_time='20080101', end_time='20090101').head()
    [39336:MainThread](2019-05-28 21:35:42,800) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:56] - provider_uri=172.23.233.89:/data2/gaochao/sync_qlib/qlib
    [39336:Thread-68](2019-05-28 21:35:42,809) INFO - Client - [client.py:28] - Connect to server ws://172.23.233.89:9710
    [39336:Thread-72](2019-05-28 21:35:43,489) INFO - Client - [client.py:31] - Disconnect from server!
    Opening /data/csdesign/qlib/cache/d239a3b191daa9a5b1b19a59beb47b33 in read-only mode
    Out[5]:
                               $close
    instrument datetime
    SH600000   2008-01-02  119.079704
               2008-01-03  113.120125
               2008-01-04  117.878860
               2008-01-07  124.505539
               2008-01-08  125.395004





客户端将把 `provider_uri` 中的数据挂载到 `mount_path`。然后服务器和客户端将通过flask通信，并使用这个NFS传输数据。


如果您有本地qlib数据文件，并且想要使用离线数据而不是通过客户端服务器框架在线使用数据。
这也可以通过特定配置实现。
您可以创建这样一个配置文件 `client_config_local.yml`

.. code-block:: YAML

    provider_uri: /data/csdesign/qlib
    calendar_provider: 'LocalCalendarProvider'
    instrument_provider: 'LocalInstrumentProvider'
    feature_provider: 'LocalFeatureProvider'
    expression_provider: 'LocalExpressionProvider'
    dataset_provider: 'LocalDatasetProvider'
    provider: 'LocalProvider'
    dataset_cache: 'SimpleDatasetCache'
    local_cache_path: '~/.cache/qlib/'

`provider_uri` 是您本地数据的目录。

.. code-block:: python

   >>> import qlib
   >>> qlib.init_from_yaml_conf('client_config_local.yml')
   >>> from qlib.data import D
   >>> D.features(['SH600001'], ['$close'], start_time='20180101', end_time='20190101').head()
    21232:MainThread](2019-05-29 10:16:05,066) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [21232:MainThread](2019-05-29 10:16:05,066) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [21232:MainThread](2019-05-29 10:16:05,067) INFO - Initialization - [__init__.py:56] - provider_uri=/data/csdesign/qlib
    Out[9]:
                              $close
    instrument datetime
    SH600001   2008-01-02  21.082111
               2008-01-03  23.195362
               2008-01-04  23.874615
               2008-01-07  24.880930
               2008-01-08  24.277143

限制条件
-----------
1. 客户端-服务器模块下的以下API可能不如旧版离线API速度快。
    - Cal.calendar
    - Inst.list_instruments
2. 在客户端-服务器框架机制下，参数为`0`的滚动操作表达式可能无法正确更新。

API
***

客户端基于 `python-socketio <https://python-socketio.readthedocs.io>`_ 构建，这是一个支持Python语言WebSocket客户端的框架。客户端只能提出请求并接收结果，不包含任何计算过程。

类
-----

.. automodule:: qlib.data.client
