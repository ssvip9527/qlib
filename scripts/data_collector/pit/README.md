# 收集时点数据

> *请注意，本数据来源于[baostock](http://baostock.com)，数据可能并不完美。如果用户拥有高质量数据集，建议自行准备数据。更多信息请参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## 需求说明

```bash
pip install -r requirements.txt
```

## 数据收集


### 下载季度中国市场数据

```bash
cd qlib/scripts/data_collector/pit/
# 从baostock.com下载数据
python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly
```

下载所有股票数据非常耗时。如果您只想对少数几只股票进行快速测试，可以运行以下命令
```bash
python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly --symbol_regex "^(600519|000725).*"
```


### 数据标准化
```bash
python collector.py normalize_data --interval quarterly --source_dir ~/.qlib/stock_data/source/pit --normalize_dir ~/.qlib/stock_data/source/pit_normalized
```



### 将数据转储为PIT格式

```bash
cd qlib/scripts
python dump_pit.py dump --csv_path ~/.qlib/stock_data/source/pit_normalized --qlib_dir ~/.qlib/qlib_data/cn_data --interval quarterly
```
