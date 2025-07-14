# 时间融合Transformer基准测试
## 来源
**参考文献**：Lim, Bryan, 等. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." arXiv预印本 arXiv:1912.09363 (2019).

**GitHub**：https://github.com/google-research/google-research/tree/master/tft

## 运行工作流
用户可以按照``workflow_by_code_tft.py``来运行基准测试。

### 注意事项
1. 请注意，此脚本仅支持`Python 3.6 - 3.7`。
2. 如果您机器上的CUDA版本不是10.0，请记得在您的机器上运行以下命令：`conda install anaconda cudatoolkit=10.0` 和 `conda install cudnn`。
3. 模型必须在GPU上运行，否则会引发错误。
4. 新数据集应在``data_formatters``中注册，详细信息请访问来源。
