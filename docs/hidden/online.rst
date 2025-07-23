.. _online:

Online
======
.. currentmodule:: qlib在线服务
======

简介
------------

欢迎使用在线服务模块，该模块模拟了使用我们的模型和策略进行实际交易的场景。

就像Qlib中的Estimator和其他模块一样，您需要通过配置文件来确定参数，
在这个模块中，您需要在文件夹中添加一个账户来进行模拟。然后在每个交易日，
该模块将使用最新的信息为您的账户进行交易，
您可以随时使用我们定义的API查看性能。

每个账户都将经历以下过程，'pred_date'表示您预测交易后目标
仓位的日期，而'trade_date'是您进行交易的日期。

- 生成订单列表（预测日期）
- 执行订单列表（交易日期）
- 更新账户（交易日期）

同时，您可以创建一个账户并使用此模块来测试其在一段时期内的表现。

- 模拟（开始日期，结束日期）

此模块需要将您的账户保存在一个文件夹中，模型和策略将保存为pickle文件，
仓位和报告将保存为excel文件。
文件结构可以在fileStruct_中查看。


示例
-------

让我们通过一个示例来说明，

.. note:: 请确保您已安装最新版本的 `qlib`。

如果您想使用 `qlib` 提供的模型和数据，您只需要按照以下步骤操作。

首先，编写一个简单的配置文件，如下所示：

.. code-block:: YAML

    strategy:
        class: TopkAmountStrategy
        module_path: qlib.contrib.strategy
        args:
            market: csi500
            trade_freq: 5

    model:
        class: ScoreFileModel
        module_path: qlib.contrib.online.online_model
        args:
            loss: mse
            model_path: ./model.bin

    init_cash: 1000000000

然后我们可以使用以下命令创建一个文件夹，并在2017-01-01到2018-08-01期间进行交易。

.. code-block:: bash

    online simulate -id v-test -config ./config/config.yaml -exchange_config ./config/exchange.yaml -start 2017-01-01 -end 2018-08-01 -path ./user_data/

开始日期（2017-01-01）是用户的添加日期，也是第一个预测日期，
结束日期（2018-08-01）是最后的交易日期。您可以使用 "`online generate -date 2018-08-02...`"
命令在下一个交易日继续生成订单列表。

如果您的账户保存在 "./user_data/"，您可以通过以下命令将您的账户性能与基准进行比较

.. code-block:: bash

    >> online show -id v-test -path ./user_data/ -bench SH000905

    ...
    Result of porfolio:
                                                      risk
    excess_return_without_cost mean               0.000605
                               std                0.005481
                               annualized_return  0.152373
                               information_ratio  1.751319
                               max_drawdown      -0.059055
    excess_return_with_cost    mean               0.000410
                               std                0.005478
                               annualized_return  0.103265
                               information_ratio  1.187411
                               max_drawdown      -0.075024


这里 'SH000905' 代表中证500，'SH000300' 代表沪深300

账户管理
-------------------

所有通过 `online` 模块处理的账户都应保存在文件夹中。您可以使用定义的命令来管理账户。

- add an new account
    This will add an new account with user_id='v-test', add_date='2019-10-15' in ./user_data.

    .. code-block:: bash

        >> online add_user -id {user_id} -config {config_file} -path {folder_path} -date {add_date}
        >> online add_user -id v-test -config config.yaml -path ./user_data/ -date 2019-10-15

- remove an account
    .. code-block:: bash

        >> online remove_user -id {user_id} -path {folder_path}
        >> online remove_user -id v-test -path ./user_data/

- show the performance
    Here benchmark indicates the baseline is to be compared with yours.

    .. code-block:: bash

        >> online show -id {user_id} -path {folder_path} -bench {benchmark}
        >> online show -id v-test -path ./user_data/ -bench SH000905

所有参数'date'的默认值都是交易日期
（如果今天是交易日期且信息已在`qlib`中更新，则为今天）。

'generate'和'update'将检查输入日期是否有效，以下3个过程应
在每个交易日调用。

- 生成订单列表
    在交易日期生成订单列表，并将其保存在{folder_path}/{user_id}/temp/中作为json文件。

    .. code-block:: bash

        >> online generate -date {date} -path {folder_path}
        >> online generate -date 2019-10-16 -path ./user_data/

- 执行订单列表
    在交易日期执行订单列表并在{folder_path}/{user_id}/temp/中生成交易结果

    .. code-block:: bash

        >> online execute -date {date} -exchange_config {exchange_config_path} -path {folder_path}
        >> online execute -date 2019-10-16 -exchange_config ./config/exchange.yaml -path ./user_data/

    一个简单的交易所配置文件可以是

    .. code-block:: yaml

        open_cost: 0.003
        close_cost: 0.003
        limit_threshold: 0.095
        deal_price: vwap


- 更新账户
    在交易日期更新"{folder_path}/"中的账户

    .. code-block:: bash

        >> online update -date {date} -path {folder_path}
        >> online update -date 2019-10-16 -path ./user_data/

API
---

所有这些操作都基于 `qlib.contrib.online.operator` 中定义的接口

.. automodule:: qlib.contrib.online.operator

.. _fileStruct:

文件结构
--------------

'user_data' 表示文件夹的根目录。
粗体名称表示文件夹，否则为文档。

.. code-block:: yaml

    {user_folder}
    │   users.csv: (Init date for each users)
    │
    └───{user_id1}: (users' sub-folder to save their data)
    │   │   position.xlsx
    │   │   report.csv
    │   │   model_{user_id1}.pickle
    │   │   strategy_{user_id1}.pickle
    │   │
    │   └───score
    │   │   └───{YYYY}
    │   │       └───{MM}
    │   │           │   score_{YYYY-MM-DD}.csv
    │   │
    │   └───trade
    │       └───{YYYY}
    │           └───{MM}
    │               │   orderlist_{YYYY-MM-DD}.json
    │               │   transaction_{YYYY-MM-DD}.csv
    │
    └───{user_id2}
    │   │   position.xlsx
    │   │   report.csv
    │   │   model_{user_id2}.pickle
    │   │   strategy_{user_id2}.pickle
    │   │
    │   └───score
    │   └───trade
    ....


配置文件
------------------

在`online`中使用的配置文件应包含模型和策略信息。

关于模型
~~~~~~~~~~~~~~~

首先，您的配置文件需要有一个关于模型的字段，
这个字段及其内容决定了我们在预测日期生成分数时使用的模型。

以下是ScoreFileModel的两个示例，一个模型读取分数文件并在交易日期返回分数。

.. code-block:: YAML

     model:
        class: ScoreFileModel
        module_path: qlib.contrib.online.OnlineModel
        args:
            loss: mse

.. code-block:: YAML

     model:
        class: ScoreFileModel
        module_path: qlib.contrib.online.OnlineModel
        args:
            score_path: <your score path>

如果您的模型不属于上述模型，您需要手动编写您的模型。
您的模型应该是'qlib.contfib.model'中定义的模型的子类。并且必须
包含在`online`模块中使用的2个方法。


关于策略
~~~~~~~~~~~~~~~~~~

您需要定义用于在预测日期生成订单列表的策略。

以下是TopkAmountStrategy的两个示例

.. code-block:: YAML

    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy.strategy
        args:
            topk: 100
            n_drop: 10

生成的文件
---------------

'online_generate'命令将在{folder_path}/{user_id}/temp/创建订单列表，
文件名为orderlist_{YYYY-MM-DD}.json，YYYY-MM-DD是这些订单将被执行的日期。

json文件的格式如下

.. code-block:: python

    {
        'sell': {
                {'$stock_id1': '$amount1'},
                {'$stock_id2': '$amount2'}, ...
                },
        'buy': {
                {'$stock_id1': '$amount1'},
                {'$stock_id2': '$amount2'}, ...
                }
    }

然后在执行订单列表后（无论是通过'online_execute'还是其他执行器），
一个交易文件也将在{folder_path}/{user_id}/temp/中创建。
