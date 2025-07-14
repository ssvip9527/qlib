
- [下载Qlib数据](#Download-Qlib-Data)
  - [下载中国市场数据](#Download-CN-Data)
  - [下载美国市场数据](#Download-US-Data)
  - [下载中国市场简化数据](#Download-CN-Simple-Data)
  - [帮助信息](#Help)
- [在Qlib中使用](#Using-in-Qlib)
  - [美国市场数据](#US-data)
  - [中国市场数据](#CN-data)


## 下载Qlib数据


### 下载中国市场数据

```bash
# 日线数据
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 1分钟数据（非高频策略可选）
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
```

### 下载美国市场数据


```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### 下载中国市场简化数据

```bash
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 帮助信息

```bash
python get_data.py qlib_data --help
```

## 在Qlib中使用
> 更多信息请参考: https://qlib.readthedocs.io/en/latest/start/initialization.html


### 美国市场数据

> 需先下载数据: [下载美国市场数据](#Download-US-Data)

```python
import qlib
from qlib.config import REG_US
provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)
```

### 中国市场数据

> 需先下载数据: [下载中国市场数据](#Download-CN-Data)

```python
import qlib
from qlib.constant import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

## 使用社区贡献数据
社区贡献的Qlib数据版本: [crowd sourced version of qlib data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```
