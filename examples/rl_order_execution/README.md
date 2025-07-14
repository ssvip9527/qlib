# 订单执行的强化学习示例

本文件夹包含用于订单执行场景的强化学习（RL）工作流示例，包括训练工作流和回测工作流。

## 数据处理

### 获取数据

```
python -m qlib.run.get_data qlib_data qlib_data --target_dir ./data/bin --region hs300 --interval 5min
```

### 生成Pickle格式数据

要运行本示例中的代码，我们需要pickle格式的数据。为此，请运行以下命令（可能需要几分钟完成）：

[//]: # (TODO: 建议实现`Dataset`和`DataHandler`的不同子类，而非使用不同格式转储数据框（如`qlib/contrib/data/highfreq_provider.py`中的`_gen_dataset`和`_gen_day_dataset`）。这将使工作流更清晰，接口更一致，并将所有复杂性转移到子类中。)

```
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
python scripts/gen_training_orders.py
python scripts/merge_orders.py
```

完成后，`data/`目录下的结构应如下所示：

```
data
├── bin
├── orders
└── pickle
```

## 训练

每个训练任务由配置文件指定。任务`TASKNAME`的配置文件为`exp_configs/train_TASKNAME.yml`。本示例提供两个训练任务：

- **PPO**：IJCAI 2020论文《[基于近端策略优化的端到端最优交易执行框架](https://www.ijcai.org/proceedings/2020/0627.pdf)》中提出的方法。
- **OPDS**：AAAI 2021论文《[基于Oracle策略蒸馏的通用订单执行交易](https://arxiv.org/abs/2103.10860)》中提出的方法。

这两种方法的主要区别在于它们的奖励函数。详情请参见它们的配置文件。

以OPDS为例，要运行训练工作流，请执行：

```
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest
```

指标、日志和检查点将存储在`outputs/opds`目录下（由`exp_configs/train_opds.yml`配置）。

## 回测

训练工作流完成后，训练好的模型可用于回测工作流。仍以OPDS为例，训练完成后，模型的最新检查点可在`outputs/opds/checkpoints/latest.pth`找到。要运行回测工作流：

1. 取消`exp_configs/train_opds.yml`中`weight_file`参数的注释（默认情况下是注释的）。虽然可以不设置检查点运行回测工作流，但这将导致模型结果随机初始化，因此没有意义。
2. 运行`python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml`。

回测结果存储在`outputs/checkpoints/backtest_result.csv`中。

除OPDS和PPO外，我们还提供TWAP（[时间加权平均价格](https://en.wikipedia.org/wiki/Time-weighted_average_price)）作为弱基准。TWAP的配置文件为`exp_configs/backtest_twap.yml`。

### 回测与训练 pipeline 测试之间的差异

值得注意的是，回测过程的结果可能与训练期间测试过程的结果不同。
这是因为训练和回测期间使用不同的模拟器来模拟市场条件。
在训练pipeline中，出于效率考虑，使用了名为`SingleAssetOrderExecutionSimple`的简化模拟器。
`SingleAssetOrderExecutionSimple`对交易量没有限制。
无论订单金额多少，都可以完全执行。
然而，在回测期间，使用了更真实的模拟器`SingleAssetOrderExecution`。
它考虑了更真实场景中的实际约束（例如，交易量必须是最小交易单位的倍数）。
因此，回测期间实际执行的订单金额可能与预期执行的金额不同。

如果您希望获得与训练pipeline中测试期间完全相同的结果，可以仅运行训练pipeline的回测阶段。
为此：
- 修改训练配置。添加您要使用的检查点路径（示例如下）。
- 运行`python -m qlib.rl.contrib.train_onpolicy --config_path PATH/TO/CONFIG --run_backtest --no_training`

```yaml
...
policy:
  class: PPO  # PPO, DQN
  kwargs:
    lr: 0.0001
    weight_file: PATH/TO/CHECKPOINT
  module_path: qlib.rl.order_execution.policy
...
```

## 基准测试（待完成）

为了准确评估使用强化学习算法的模型性能，最好多次运行实验并计算所有试验的平均性能。然而，由于模型训练耗时，这并不总是可行的。另一种方法是每个训练任务只运行一次，选择验证性能最高的10个检查点来模拟多次试验。在本示例中，我们使用“价格优势（PA）”作为选择这些检查点的指标。这10个检查点在测试集上的平均性能如下：

| **模型**                    | **PA均值±标准差**     |
|-----------------------------|-----------------------|
| OPDS（使用PPO策略）         |  0.4785 ± 0.7815      |
| OPDS（使用DQN策略）         | -0.0114 ± 0.5780      |
| PPO                         | -1.0935 ± 0.0922      |
| TWAP                        |   ≈ 0.0 ± 0.0         |

上表还包括TWAP作为基于规则的基准。TWAP的理想PA应为0.0，然而在本示例中，订单执行分为两步：首先，订单在每个半小时内平均拆分，然后在每个半小时内的每五分钟拆分。由于当天最后五分钟禁止交易，这种方法可能与全天传统TWAP略有不同（因为最后一个“半小时”缺少5分钟）。因此，TWAP的PA可视为接近0.0的数值。要验证这一点，您可以运行TWAP回测并查看结果。
