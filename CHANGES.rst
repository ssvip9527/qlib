更新日志
=========
您可以在此处查看每个 QLib 版本之间的完整变更列表。

版本 0.1.0
-------------
QLib 库的初始发布。

版本 0.1.1
-------------
性能优化，增加更多功能和算子。

版本 0.1.2
-------------
- 支持算子语法。现在 ``High() - Low()`` 等价于 ``Sub(High(), Low())``。
- 增加更多技术指标。

版本 0.1.3
-------------
修复 bug 并增加标的过滤机制。

版本 0.2.0
-------------
- 重新设计 ``LocalProvider`` 数据库格式以提升性能。
- 支持以字符串字段加载特征。
- 增加数据库构建脚本。
- 更多算子和技术指标。

版本 0.2.1
-------------
- 支持注册用户自定义 ``Provider``。
- 支持字符串格式的算子，例如 ``['Ref($close, 1)']`` 是有效的字段格式。
- 支持 ``$some_field`` 格式的动态字段，现有如 ``Close()`` 的字段未来可能弃用。

版本 0.2.2
-------------
- 增加 ``disk_cache`` 用于复用特征（默认启用）。
- 增加 ``qlib.contrib`` 用于实验性模型构建与评估。


版本 0.2.3
-------------
- 增加 ``backtest`` 回测模块
- 将策略、账户、持仓、交易所与回测模块解耦

版本 0.2.4
-------------
- 增加 ``profit attribution`` 收益归因模块
- 增加 ``rick_control`` 和 ``cost_control`` 策略

版本 0.3.0
-------------
- 增加 ``estimator`` 模块

版本 0.3.1
-------------
- 增加 ``filter`` 模块

版本 0.3.2
-------------
- 增加真实价格交易，若数据集中的 ``factor`` 字段不完整，则使用 ``adj_price`` 进行交易
- 重构 ``handler``、``launcher``、``trainer`` 代码
- 支持在配置文件中设置 ``backtest`` 回测参数
- 修复持仓 ``amount`` 为 0 的 bug
- 修复 ``filter`` 模块的 bug

版本 0.3.3
-------------
- 修复 ``filter`` 模块的 bug

版本 0.3.4
-------------
- 支持 ``finetune model`` 微调模型
- 重构 ``fetcher`` 代码

版本 0.3.5
-------------
- 支持多标签训练，可在 ``handler`` 中提供多个标签（LightGBM 由于算法本身不支持）
- 重构 ``handler`` 代码，不再使用 dataset.py，可在 ``feature_label_config`` 中自定义标签和特征
- Handler 仅提供 DataFrame，``trainer`` 和 model.py 也只接收 DataFrame
- 更改 ``split_rolling_data``，现在按市场日历滚动数据，而非普通日期
- 将部分日期配置从 ``handler`` 移至 ``trainer``

版本 0.4.0
-------------
- 增加 `data` 包，包含所有与数据相关的代码
- 重构数据提供者结构
- 创建用于数据集中管理的服务器 `qlib-server <https://amc-msra.visualstudio.com/trading-algo/_git/qlib-server>`_
- 增加与服务器配合使用的 `ClientProvider`
- 增加可插拔缓存机制
- 增加递归回溯算法以检查表达式最远引用日期

.. note::
    ``D.instruments`` 函数不支持 ``start_time``、``end_time`` 和 ``as_list`` 参数，若需获取旧版本 ``D.instruments`` 的结果，可如下操作：

    >>> from qlib.data import D
    >>> instruments = D.instruments(market='csi500')
    >>> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)

版本 0.4.1
-------------
- 增加对 Windows 的支持
- 修复 ``instruments`` 类型 bug
- 修复 ``features`` 为空导致更新失败的 bug
- 修复 ``cache`` 锁和更新 bug
- 修复同一字段使用相同缓存（原空间会新增缓存）
- 日志处理器从配置中更改
- 模型加载支持 0.4.0 及以后版本
- ``risk_analysis`` 函数的 ``method`` 参数默认值由 **ci** 改为 **si**


版本 0.4.2
-------------
- 重构 DataHandler
- 增加 ``Alpha360`` DataHandler

版本 0.4.3
-------------
- 实现在线推理与交易框架
- 重构回测与策略模块接口

版本 0.4.4
-------------
- 优化缓存生成性能
- 增加报告模块
- 修复离线使用 ``ServerDatasetCache`` 时的 bug
- 之前版本 ``long_short_backtest`` 存在 long_short 为 ``np.nan`` 的情况，当前 ``0.4.4`` 版本已修复，因此 ``long_short_backtest`` 结果与之前版本不同
- ``risk_analysis`` 函数在 ``0.4.2`` 版本中 ``N`` 为 ``250``，在 ``0.4.3`` 及以后为 ``252``，因此 ``0.4.2`` 比 ``0.4.3`` 回测结果小 ``0.002122``，两版本回测结果略有差异
- 重构回测函数参数
    - **注意**：
      - topk margin 策略的默认参数已更改，如需获得与旧版本一致的回测结果，请显式传递参数
      - TopkWeightStrategy 行为略有变化，会尝试卖出超过 ``topk`` 的股票（TopkAmountStrategy 回测结果保持不变）
- Topk Margin 策略支持保证金比例机制

版本 0.4.5
-------------
- 客户端和服务器均支持多内核实现
    - 支持客户端跳过数据集缓存的新数据加载方式
    - 默认数据集方法由单内核实现改为多内核实现
- 通过优化相关模块加速高频数据读取
- 支持通过 dict 写配置文件的新方法

版本 0.4.6
-------------
- 修复部分 bug
    - `Version 0.4.5` 的默认配置对日频数据不友好
    - TopkWeightStrategy 在 `WithInteract=True` 时回测报错

版本 0.5.0
-------------
- 首个开源版本
    - 优化文档和代码
    - 增加基线模型
    - 公共数据爬虫

版本 0.8.0
-------------
- 回测模块大幅重构
    - 支持嵌套决策执行框架
    - 日内交易有大量变更，难以一一列举，主要变化包括：
        - 交易限制更为精确：
            - `旧版本 <https://github.com/ssvip9527/qlib/blob/v0.7.2/qlib/contrib/backtest/exchange.py#L160>`__，多空操作共用同一动作
            - `当前版本 <https://github.com/ssvip9527/qlib/blob/7c31012b507a3823117bddcc693fc64899460b2a/qlib/backtest/exchange.py#L304>`__，多空操作的交易限制不同
        - 年化指标计算常数不同：
            - `Current version <https://github.com/ssvip9527/qlib/blob/7c31012b507a3823117bddcc693fc64899460b2a/qlib/contrib/evaluate.py#L42>`_ 使用更精确常数，优于 `previous version <https://github.com/ssvip9527/qlib/blob/v0.7.2/qlib/contrib/evaluate.py#L22>`__
        - 发布了 `A new version <https://github.com/ssvip9527/qlib/blob/7c31012b507a3823117bddcc693fc64899460b2a/qlib/tests/data.py#L17>`__ 的数据。由于 Yahoo 数据源不稳定，重新下载数据后可能不同
        - 用户可对比 `Current version <https://github.com/ssvip9527/qlib/tree/7c31012b507a3823117bddcc693fc64899460b2a/examples/benchmarks>`__ 与 `previous version <https://github.com/ssvip9527/qlib/tree/v0.7.2/examples/benchmarks>`__ 的回测结果

其它版本
--------------
请参考 `GitHub 发布说明 <https://github.com/ssvip9527/qlib/releases>`_
