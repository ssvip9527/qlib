# 简介

本示例旨在演示Qlib如何支持非固定共享频率的数据。

例如：
- 日线级价格成交量数据是固定频率数据，数据以固定频率（如日线）生成
- 订单数据是非固定频率数据，可能在任意时间点产生

为支持此类非固定频率数据，Qlib实现了基于Arctic的后端。
以下是基于此后端导入和查询数据的示例。

# 安装

请参考MongoDB的[安装文档](https://docs.mongodb.com/manual/installation/)。
当前版本的脚本默认尝试通过默认端口连接本地主机，且无需身份验证。

运行以下命令安装必要的库：
```
pip install pytest coverage gdown
pip install arctic  # 注意：pip可能无法正确解析依赖包！请确保所有依赖都已满足。
```

# 导入示例数据


1. （可选）请按照[此部分](https://github.com/microsoft/qlib#data-preparation)的第一部分获取Qlib的**1分钟高频数据**。
2. 请按照以下步骤下载示例数据：
```bash
cd examples/orderbook_data/
gdown https://drive.google.com/uc?id=15nZF7tFT_eKVZAcMFL1qPS4jGyJflH7e  # 此处可能需要代理
python ../../scripts/get_data.py _unzip --file_path highfreq_orderbook_example_data.zip --target_dir .
```

3. 请将示例数据导入到您的MongoDB中：
```bash
python create_dataset.py initialize_library  # 初始化库
python create_dataset.py import_data  # 导入数据
```

# 查询示例

导入数据后，您可以运行`example.py`来创建一些高频特征：
```bash
pytest -s --disable-warnings example.py   # 如果要运行所有示例
pytest -s --disable-warnings example.py::TestClass::test_exp_10  # 如果要运行特定示例
```


# 已知限制
尚不支持不同频率之间的表达式计算
