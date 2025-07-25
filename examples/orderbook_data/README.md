# 简介

本示例旨在演示Qlib如何支持非固定共享频率的数据。

例如：
- 日线级价格成交量数据是固定频率数据，数据以固定频率（如日线）生成
- 订单数据是非固定频率数据，可能在任意时间点产生

为支持此类非固定频率数据，Qlib实现了基于ArcticDB的后端（替代了原有的Arctic后端）。
以下是基于此后端导入和查询数据的示例。

# 安装

请参考MongoDB的[安装文档](https://docs.mongodb.com/manual/installation/)。
当前版本的脚本默认尝试通过默认端口连接本地主机，且无需身份验证。

运行以下命令安装必要的库：
```
pip install pytest coverage gdown
pip install arcticdb>=1.4.1  # 使用ArcticDB替代原有的Arctic
```

## 从Arctic迁移到ArcticDB

如果您之前使用的是基于Arctic的版本，请参考 [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) 了解如何迁移到ArcticDB。

我们提供了迁移工具脚本 [migrate_to_arcticdb.py](./migrate_to_arcticdb.py) 帮助您将数据从Arctic迁移到ArcticDB。

# 导入示例数据


1. （可选）请按照[此部分](https://github.com/ssvip9527/qlib#data-preparation)的第一部分获取Qlib的**1分钟高频数据**。
2. 请按照以下步骤下载示例数据：
```bash
cd examples/orderbook_data/
gdown https://drive.google.com/uc?id=15nZF7tFT_eKVZAcMFL1qPS4jGyJflH7e  # 此处可能需要代理
python ../../scripts/get_data.py _unzip --file_path highfreq_orderbook_example_data.zip --target_dir .
```

3. 请将示例数据导入到ArcticDB中：
```bash
python create_dataset_arcticdb.py initialize_library  # 初始化库
python create_dataset_arcticdb.py import_data  # 导入数据
```

# 查询示例

导入数据后，您可以运行`example_arcticdb.py`来创建一些高频特征：
```bash
pytest -s --disable-warnings example_arcticdb.py   # 如果要运行所有示例
pytest -s --disable-warnings example_arcticdb.py::TestClass::test_exp_10  # 如果要运行特定示例
```

# 使用ArcticDB的优势

- **无需MongoDB服务器**：ArcticDB使用本地LMDB存储或S3，无需额外的数据库服务器
- **更高性能**：ArcticDB提供更快的读写速度和更高效的数据压缩
- **更简单的安装**：通过pip安装，无需复杂配置
- **更好的兼容性**：与最新版本的pandas和numpy兼容性更好


# 已知限制
尚不支持不同频率之间的表达式计算
