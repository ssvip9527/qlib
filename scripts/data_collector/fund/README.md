# 收集基金数据

> *请注意，本数据来源于[天天基金网](https://fund.eastmoney.com/)，数据可能并不完美。如果用户拥有高质量数据集，建议自行准备数据。更多信息请参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## 需求说明

```bash
pip install -r requirements.txt
```

## 数据收集


### 中国市场数据

#### 来自东方财富网的1日数据

```bash

# 从东方财富网下载数据
python collector.py download_data --source_dir ~/.qlib/fund_data/source/cn_data --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# 数据标准化
python collector.py normalize_data --source_dir ~/.qlib/fund_data/source/cn_data --normalize_dir ~/.qlib/fund_data/source/cn_1d_nor --region CN --interval 1d --date_field_name FSRQ

# 生成二进制文件
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/fund_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/cn_fund_data --freq day --date_field_name FSRQ --include_fields DWJZ,LJJZ

```

### 使用数据

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/cn_fund_data")
df = D.features(D.instruments(market="all"), ["$DWJZ", "$LJJZ"], freq="day")
```


### 帮助信息
```bash
pythono collector.py collector_data --help
```

## 参数说明

- interval: 1d
- region: CN

## 免责声明

本项目仅供学习研究使用，不作为任何行为的指导和建议，由此而引发任何争议和纠纷，与本项目无任何关系
