# 嵌套决策执行

本工作流是回测中嵌套决策执行的示例。Qlib支持在回测中进行嵌套决策执行，这意味着用户可以在不同频率下使用不同策略进行交易决策。

## 周度投资组合生成与日线级订单执行

本工作流提供了一个示例，使用周频的DropoutTopkStrategy（基于日线频率Lightgbm模型的策略）生成投资组合，并使用日线级的SBBStrategyEMA（基于EMA规则的策略）执行订单。

### 使用方法

运行以下命令开始回测：
```bash
    python workflow.py backtest
```

运行以下命令开始收集数据：
```bash
    python workflow.py collect_data
```

## 日线级投资组合生成与分钟级订单执行

本工作流还提供了一个高频示例，使用日线频率的DropoutTopkStrategy生成投资组合，并使用分钟级的SBBStrategyEMA执行订单。

### 使用方法

运行以下命令开始回测：
```bash
    python workflow.py backtest_highfreq
```