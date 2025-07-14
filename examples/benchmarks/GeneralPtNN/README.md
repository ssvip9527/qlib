

# 简介

什么是GeneralPtNN
- 修复了之前设计中无法同时支持时间序列和表格数据的问题
- 现在您只需替换PyTorch模型结构即可运行神经网络模型。

我们提供了一个示例来展示当前设计的有效性。
- `workflow_config_gru.yaml` 与之前的结果 [GRU(Kyunghyun Cho等人)](../README.md#Alpha158-dataset) 一致
  - `workflow_config_gru2mlp.yaml` 展示了我们可以通过最少的更改将配置从时间序列数据转换为表格数据
    - 您只需更改网络和数据集类即可完成转换。
- `workflow_config_mlp.yaml` 实现了与 [MLP](../README.md#Alpha158-dataset) 类似的功能

# 待办事项

- 我们将使现有模型与当前设计保持一致。

- `workflow_config_mlp.yaml` 的结果与 [MLP](../README.md#Alpha158-dataset) 的结果不同，因为GeneralPtNN与之前的实现相比具有不同的停止方法。具体来说，GeneralPtNN根据轮次（epochs）控制训练，而之前的方法通过max_steps控制。
