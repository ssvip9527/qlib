# 简介
本文基于Qlib提供的`Meta Controller`（元控制器）组件实现了`DDG-DA`算法。

更多细节请参考论文：*DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation* [[arXiv](https://arxiv.org/abs/2201.04038)]


# 背景
在许多实际场景中，我们经常处理随时间顺序收集的流数据。由于环境的非平稳性，流数据分布可能以不可预测的方式变化，这被称为概念漂移。为了处理概念漂移，以往的方法首先检测概念漂移发生的时间和位置，然后调整模型以适应最新数据的分布。然而，在许多情况下，环境演变的一些潜在因素是可预测的，这使得对流数据未来的概念漂移趋势进行建模成为可能，而这些情况在以往的研究中尚未得到充分探索。

因此，我们提出了一种新的方法`DDG-DA`，能够有效预测数据分布的演变并提高模型性能。具体来说，我们首先训练一个预测器来估计未来的数据分布，然后利用它生成训练样本，最后在生成的数据上训练模型。

# 数据集
论文中使用的数据为私有数据。因此，我们在Qlib的公开数据集上进行实验。
尽管数据集不同，但结论保持一致。通过应用`DDG-DA`，用户可以在测试阶段看到代理模型的IC值和预测模型性能均呈上升趋势。

# 运行代码
用户可以通过以下命令尝试`DDG-DA`：
```bash
    python workflow.py run
```

默认的预测模型是`Linear`（线性模型）。用户可以在初始化`DDG-DA`时通过修改`forecast_model`参数选择其他预测模型。例如，用户可以通过以下命令尝试`LightGBM`预测模型：
```bash
    python workflow.py --conf_path=../workflow_config_lightgbm_Alpha158.yaml run
```

# 结果
Qlib公开数据集上相关方法的结果可以在[这里](../)找到

# 要求
以下是运行DDG-DA的`workflow.py`所需的最低硬件要求：
* 内存：45G
* 磁盘：4G

本示例只需配备CPU和足够RAM的Pytorch环境即可运行。
