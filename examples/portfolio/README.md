# 投资组合优化策略

## 简介

在`qlib/examples/benchmarks`中，我们提供了多种**alpha**模型用于预测股票收益。我们还使用基于简单规则的`TopkDropoutStrategy`来评估这些模型的投资表现。然而，这种策略过于简单，无法控制投资组合的风险（如相关性和波动率）。

为此，应使用基于优化的策略来平衡收益与风险。本文档将展示如何使用`EnhancedIndexingStrategy`在最小化相对于基准跟踪误差的同时最大化投资组合收益。


## 准备工作

本示例使用中国股票市场数据。

1. 准备沪深300指数权重：

   ```bash
   wget https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/csi300_weight.zip
   unzip -d ~/.qlib/qlib_data/cn_data csi300_weight.zip
   rm -f csi300_weight.zip
   ```
   注意：我们未找到公开免费的基准权重数据资源。为运行本示例，我们手动创建了此权重数据。

2. 准备风险模型数据：

   ```bash
   python prepare_riskdata.py
   ```

此处我们使用`qlib.model.riskmodel`中实现的**统计风险模型**。但强烈建议用户使用其他更高质量的风险模型：
* **基本面风险模型**，如MSCI BARRA
* [深度风险模型](https://arxiv.org/abs/2107.05201)


## 端到端工作流

您可以通过运行`qrun config_enhanced_indexing.yaml`来完成使用`EnhancedIndexingStrategy`的工作流。

在此配置中，与`qlib/examples/benchmarks/workflow_config_lightgbm_Alpha158.yaml`相比，我们主要修改了策略部分。
