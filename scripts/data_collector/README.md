# 数据收集器

## 简介

数据收集相关脚本

- yahoo: 从*Yahoo Finance*获取*美国/中国*股票数据
- fund: 从*http://fund.eastmoney.com*获取基金数据
- cn_index: 从*http://www.csindex.com.cn*获取*中国指数*，如*沪深300*/*中证100*
- us_index: 从*https://en.wikipedia.org/wiki*获取*美国指数*，如*标普500*/*纳斯达克100*/*道琼斯工业平均指数*/*标普400*
- contrib: 包含一些辅助功能的脚本


## 自定义数据收集

> 具体实现参考: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo

1. 在当前目录创建数据集代码目录
2. 添加`collector.py`文件
   - 添加收集器类:
     ```python
     CUR_DIR = Path(__file__).resolve().parent
     sys.path.append(str(CUR_DIR.parent.parent))
     from data_collector.base import BaseCollector, BaseNormalize, BaseRun
     class UserCollector(BaseCollector):
         ...
     ```
   - 添加标准化类:
     ```python
     class UserNormalzie(BaseNormalize):
         ...
     ```
   - 添加`CLI`类:
     ```python
     class Run(BaseRun):
         ...
     ```
3. 添加`README.md`文件
4. 添加`requirements.txt`文件


## 数据集说明

  |-------------|----------------------------------------------------------------------------------------------------------------|
  |------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
  | 特征        | **价格/成交量**: <br>&nbsp;&nbsp; - $close(收盘价)/$open(开盘价)/$low(最低价)/$high(最高价)/$volume(成交量)/$change(涨跌幅)/$factor(复权因子) |
  | 交易日历    | **\<freq>.txt**: <br>&nbsp;&nbsp; - day.txt(日线)<br>&nbsp;&nbsp;  - 1min.txt(1分钟线)                                  |
  | 标的列表    | **\<market>.txt**: <br>&nbsp;&nbsp; - 必需: **all.txt**(所有标的); <br>&nbsp;&nbsp;  - csi300.txt(沪深300)/csi500.txt(中证500)/sp500.txt(标普500) |

  - `特征`: 数据，**数字型**
    - 若未**复权**，则**factor=1**

### 数据依赖组件

> 为确保组件正常运行，需要以下依赖数据

  | 组件           | 所需数据                                         |
  |---------------------------------------------------|--------------------------------|
  | 数据获取       | 特征(Features)、交易日历(Calendar)、标的列表(Instrument) |
  | 回测           | **特征[价格/成交量]**, 交易日历, 标的列表          |