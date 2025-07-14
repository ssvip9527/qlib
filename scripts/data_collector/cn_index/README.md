# 沪深300/中证100/中证500指数历史成分股收集

## 需求说明

```bash
pip install -r requirements.txt
```

## 数据收集命令

```bash
# 解析工具列表，用于qlib/instruments
python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method parse_instruments

# 解析新增公司
python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method save_new_companies

# index_name支持：CSI300（沪深300）、CSI100（中证100）、CSI500（中证500）
# 帮助信息
python collector.py --help
```

