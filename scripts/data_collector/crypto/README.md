# 收集加密货币数据

> *请注意，本数据来源于[Coingecko](https://www.coingecko.com/en/api)，数据可能并不完美。如果用户拥有高质量数据集，建议自行准备数据。更多信息请参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## 需求说明

```bash
pip install -r requirements.txt
```

## 数据集使用说明
> *加密货币数据集仅支持数据检索功能，由于缺乏OHLC数据，暂不支持回测功能。*

## 数据收集


### 加密货币数据

#### 来自Coingecko的1日数据

```bash

# 从https://api.coingecko.com/api/v3/下载数据
python collector.py download_data --source_dir ~/.qlib/crypto_data/source/1d --start 2015-01-01 --end 2021-11-30 --delay 1 --interval 1d

# 数据标准化
python collector.py normalize_data --source_dir ~/.qlib/crypto_data/source/1d --normalize_dir ~/.qlib/crypto_data/source/1d_nor --interval 1d --date_field_name date

# 生成二进制文件
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/crypto_data/source/1d_nor --qlib_dir ~/.qlib/qlib_data/crypto_data --freq day --date_field_name date --include_fields prices,total_volumes,market_caps

```

### 使用数据

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/crypto_data")
df = D.features(D.instruments(market="all"), ["$prices", "$total_volumes","$market_caps"], freq="day")
```


### 帮助信息
```bash
python collector.py collector_data --help
```

## 参数说明

- interval: 1d
- delay: 1
