# 获取期货交易日

> 将使用`D.calendar(future=True)`

## 需求说明

```bash
pip install -r requirements.txt
```

## 数据收集

```bash
# 解析工具列表，用于qlib/instruments
python future_trading_date_collector.py --qlib_dir ~/.qlib/qlib_data/cn_data --freq day
```

## 参数说明

- qlib_dir: qlib数据目录
- freq: 取值范围为[`day`, `1min`]，默认值为`day`



