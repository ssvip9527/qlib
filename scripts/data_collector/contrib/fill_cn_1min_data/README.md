# 使用日线数据填充1分钟数据中缺失的标的


## 需求说明

```bash
pip install -r requirements.txt
```

## 填充1分钟数据

```bash
python fill_cn_1min_data.py --data_1min_dir ~/.qlib/csv_data/cn_data_1min --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
```

## 参数说明

- data_1min_dir: csv数据目录
- qlib_data_1d_dir: qlib数据目录
- max_workers: `ThreadPoolExecutor(max_workers=max_workers)`，默认值为*16*
- date_field_name: 日期字段名称，默认值为*date*
- symbol_field_name: 标的代码字段名称，默认值为*symbol*

