# 从 Arctic 迁移到 ArcticDB 指南

## 背景

Arctic 库已进入维护模式，开发已迁移到 ArcticDB。ArcticDB 是 Arctic 的继任者，于2023年3月发布，是一个高性能、无服务器的 DataFrame 数据库，专为 Python 数据科学生态系统构建。

## 主要优势

- **更高性能**：ArcticDB 提供更快的读写速度和更高效的数据压缩
- **无服务器架构**：不需要额外的 MongoDB 服务器，可以使用本地 LMDB 存储或 S3
- **更简单的安装**：通过 `pip install arcticdb` 即可安装，无需复杂配置
- **更好的兼容性**：与最新版本的 pandas 和 numpy 兼容性更好

## 安装 ArcticDB

```bash
pip install arcticdb>=1.4.1
```

或者通过项目的依赖管理安装：

```bash
pip install -e .
```

## API 差异

### 连接方式

**Arctic**:
```python
from arctic import Arctic
arctic = Arctic('mongodb://localhost:27017')
```

**ArcticDB**:
```python
import arcticdb as adb
arctic = adb.Arctic('lmdb:///tmp/arcticdb')  # 本地存储
# 或
arctic = adb.Arctic('s3://ENDPOINT:BUCKET?aws_auth=true')  # S3存储
```

### 库管理

**Arctic**:
```python
arctic.initialize_library('library_name', lib_type=CHUNK_STORE)
```

**ArcticDB**:
```python
arctic.create_library('library_name')
```

### 读取数据

**Arctic**:
```python
df = arctic['library_name'].read('symbol', columns=['column1'], chunk_range=(start_time, end_time))
```

**ArcticDB**:
```python
df = arctic['library_name'].read('symbol', columns=['column1'], date_range=(start_time, end_time))
```

## 数据迁移

对于小型数据集，最简单的迁移方法是从 Arctic 读取数据，然后写入 ArcticDB：

```python
# 从 Arctic 读取
from arctic import Arctic as OldArctic
old_arctic = OldArctic('mongodb://localhost:27017')
old_lib = old_arctic['library_name']
symbols = old_lib.list_symbols()

# 写入 ArcticDB
import arcticdb as adb
new_arctic = adb.Arctic('lmdb:///tmp/arcticdb')
if 'library_name' not in new_arctic.list_libraries():
    new_arctic.create_library('library_name')
new_lib = new_arctic['library_name']

# 迁移每个符号
for symbol in symbols:
    try:
        # 读取所有数据
        data = old_lib.read(symbol)
        # 写入 ArcticDB
        new_lib.write(symbol, data)
        print(f"已迁移: {symbol}")
    except Exception as e:
        print(f"迁移 {symbol} 失败: {str(e)}")
```

## 在 Qlib 中使用 ArcticDB

我们已经提供了 `ArcticDBFeatureProvider` 类来替代原有的 `ArcticFeatureProvider`。使用方法如下：

```python
import qlib

qlib.init(
    provider_uri="~/.qlib/qlib_data/yahoo_cn_1min",
    feature_provider={
        "class": "ArcticDBFeatureProvider",
        "module_path": "qlib.contrib.data.arcticdb_data",
        "kwargs": {"uri": "lmdb:///tmp/arcticdb"},
    },
)
```

## 注意事项

1. ArcticDB 针对数值型数据集进行了优化，但不能高效存储自定义 Python 对象
2. ArcticDB 的生产环境使用需要付费许可，但开发和测试是免费的
3. 迁移大型数据集时，建议分批进行，以避免内存问题

## 参考资料

- [ArcticDB 官方文档](https://docs.arcticdb.io/)
- [ArcticDB GitHub 仓库](https://github.com/man-group/ArcticDB)