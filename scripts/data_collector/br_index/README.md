# iBOVESPA指数历史成分股收集

## 需求说明

- 从`requirements.txt`文件安装依赖库

    ```bash
    pip install -r requirements.txt
    ```
- `requirements.txt`文件使用python3.8生成

## 关于ibovespa (IBOV)指数，我们提供以下功能：

<hr/>

### 方法 `get_new_companies`

#### <b>指数起始日期</b>

- ibovespa指数始于1968年1月2日（[维基百科](https://en.wikipedia.org/wiki/%C3%8Dndice_Bovespa)）。为了在`bench_start_date(self)`方法中使用此起始日期，必须满足两个条件：
    1) 用于下载巴西股票（B3）历史价格的API必须保留自1968年1月2日以来的历史数据

    2) 某个网站或API必须从该日期开始提供历史指数成分股信息，即用于构建指数的公司名单

    因此，`collector.py`中的`bench_start_date(self)`方法使用`pd.Timestamp("2003-01-03")`作为起始日期，原因如下

    1) 已找到的最早ibov成分股数据来自2003年第一季度。有关该成分股的更多信息可参见以下部分

    2) Yahoo Finance（用于下载股票历史价格的库之一）从该日期开始保留数据

- 在`get_new_companies`方法中，实现了获取每只ibovespa成分股在Yahoo Finance中可追溯的起始日期的逻辑

#### <b>代码逻辑</b>

代码通过网页抓取B3[网站](https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br)上的信息，该网站记录了当前ibovespa指数的成分股构成 

本可以使用`request`和`Beautiful Soup`等其他方法，但该网站使用内部脚本加载成分股表格，导致表格显示存在延迟
为解决此问题，我们使用`selenium`来下载成分股数据

此外，从selenium脚本下载的数据经过预处理，以保存为`scripts/data_collector/index.py`所规定的`csv`格式

<hr/>

### 方法 `get_changes` 

尚未找到合适的ibovespa历史成分股数据源。目前使用了[该仓库](https://github.com/igor17400/IBOV-HCI)提供的信息，但其仅包含2003年第一季度至2021年第三季度的数据

借助该参考数据，可以按季度和年度比较指数成分股变化，并生成记录每季度和每年新增及移除股票的文件

<hr/>

### 数据收集命令

```bash
# 解析工具列表，用于qlib/instruments
python collector.py --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method parse_instruments

# parse new companies
python collector.py --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method save_new_companies
```

