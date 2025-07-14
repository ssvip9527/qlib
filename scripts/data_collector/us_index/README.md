# 纳斯达克100/标普500/标普400/道琼斯工业平均指数历史成分股收集

## 需求说明

```bash
pip install -r requirements.txt
```

## 数据收集命令

```bash
# 解析工具列表，用于qlib/instruments
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments

# 解析新增公司
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method save_new_companies

# index_name支持：SP500（标普500）、NASDAQ100（纳斯达克100）、DJIA（道琼斯工业平均指数）、SP400（标普400）
# 帮助信息
python collector.py --help
```

