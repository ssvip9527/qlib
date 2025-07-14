
- [Collector Data](#collector-data)
  - [Get Qlib data](#get-qlib-databin-file)
  - [Collector *YahooFinance* data to qlib](#collector-yahoofinance-data-to-qlib)
  - [Automatic update of daily frequency data](#automatic-update-of-daily-frequency-datafrom-yahoo-finance)
- [Using qlib data](#using-qlib-data)


# Collect Data From Yahoo Finance

> *Please pay **ATTENTION** that the data is collected from [Yahoo Finance](https://finance.yahoo.com/lookup) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

**注意**: 雅虎财经已阻止来自中国的访问。如果您想使用雅虎数据爬虫，请更换网络。

>  **异常数据示例**

- [SH000661](https://finance.yahoo.com/quote/000661.SZ/history?period1=1558310400&period2=1590796800&interval=1d&filter=history&frequency=1d)
- [SZ300144](https://finance.yahoo.com/quote/300144.SZ/history?period1=1557446400&period2=1589932800&interval=1d&filter=history&frequency=1d)

We have considered **STOCK PRICE ADJUSTMENT**, but some price series seem still very abnormal.

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

### Get Qlib data(`bin file`)
  > `qlib-data` from *YahooFinance*, is the data that has been dumped and can be used directly in `qlib`.
  > This ready-made qlib-data is not updated regularly. If users want the latest data, please follow [these steps](#collector-yahoofinance-data-to-qlib) download the latest data. 

  - get data: `python scripts/get_data.py qlib_data`
  - parameters:
    - `target_dir`: save dir, by default *~/.qlib/qlib_data/cn_data*
    - `version`: dataset version, value from [`v1`, `v2`], by default `v1`
      - `v2` end date is *2021-06*, `v1` end date is *2020-09*
      - If users want to incrementally update data, they need to use yahoo collector to [collect data from scratch](#collector-yahoofinance-data-to-qlib).
      - **the [benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks) for qlib use `v1`**, *由于雅虎财经对历史数据的访问不稳定，`v2`和`v1`之间存在一些差异*
    - `interval`: `1d` or `1min`, by default `1d`
    - `region`: `cn` or `us` or `in`, by default `cn`
    - `delete_old`: delete existing data from `target_dir`(*features, calendars, instruments, dataset_cache, features_cache*), value from [`True`, `False`], by default `True`
    - `exists_skip`: traget_dir data already exists, skip `get_data`, value from [`True`, `False`], by default `False`
  - examples:
    ```bash
    # cn 1d
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
    # cn 1min
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
    # us 1d
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us --interval 1d
    ```

### Collector *YahooFinance* data to qlib
> 收集 *YahooFinance* 数据并 *转储* 为 `qlib` 格式。
> 如果上述现成数据无法满足用户需求，用户可以按照本部分内容爬取最新数据并转换为 qlib 数据格式。
  1. 下载数据到 csv：`python scripts/data_collector/yahoo/collector.py download_data`
     
这将从 Yahoo 下载最高价、最低价、开盘价、收盘价、复权收盘价等原始数据到本地目录，每个股票代码对应一个文件。

     - parameters:
          - `source_dir`: save the directory
          - `interval`: `1d` or `1min`, by default `1d`
            > **由于雅虎财经API的限制，`1min`频率数据仅能获取最近一个月的数据**
          - `region`: `CN` or `US` or `IN` or `BR`, by default `CN`
          - `delay`: `time.sleep(delay)`, by default *0.5*
          - `start`: start datetime, by default *"2000-01-01"*; *closed interval(including start)*
          - `end`: end datetime, by default `pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))`; *open interval(excluding end)*
          - `max_workers`: get the number of concurrent symbols, it is not recommended to change this parameter in order to maintain the integrity of the symbol data, by default *1*
          - `check_data_length`: 检查每个股票代码的数据行数，默认值为`None`
            > 如果`len(symbol_df) < check_data_length`，将重新获取数据，重新获取次数由`max_collector_count`参数决定
          - `max_collector_count`: 失败股票代码的重试次数，默认值为2
     - examples:
          ```bash
          # cn 1d data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region CN
          # cn 1min data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data_1min --delay 1 --interval 1min --region CN

          # us 1d data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region US
          # us 1min data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data_1min --delay 1 --interval 1min --region US

          # in 1d data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/in_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region IN
          # in 1min data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/in_data_1min --delay 1 --interval 1min --region IN

          # br 1d data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/br_data --start 2003-01-03 --end 2022-03-01 --delay 1 --interval 1d --region BR
          # br 1min data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/br_data_1min --delay 1 --interval 1min --region BR
          ```
  2. 数据标准化：`python scripts/data_collector/yahoo/collector.py normalize_data`
     
此步骤将：
1. 使用复权收盘价对最高价、最低价、收盘价、开盘价进行标准化处理。
2. 对最高价、最低价、收盘价、开盘价进行归一化，使第一个有效交易日的收盘价为 1。 

     - parameters:
          - `source_dir`: csv directory
          - `normalize_dir`: result directory
          - `max_workers`: number of concurrent, by default *1*
          - `interval`: `1d` or `1min`, by default `1d`
            > if **`interval == 1min`**, `qlib_data_1d_dir` cannot be `None`
          - `region`: `CN` or `US` or `IN`, by default `CN`
          - `date_field_name`: csv文件中标识时间的列名，默认值为`date`
          - `symbol_field_name`: csv文件中标识股票代码的列名，默认值为`symbol`
          - `end_date`: if not `None`, normalize the last date saved (*including end_date*); if `None`, it will ignore this parameter; by default `None`
          - `qlib_data_1d_dir`: qlib directory(1d data)
            ```
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;
        
                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --end_date <end_date>
                or:
                    download 1d data from YahooFinance
            
            ```
      - examples:
        ```bash
        # normalize 1d cn
        python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_data --normalize_dir ~/.qlib/stock_data/source/cn_1d_nor --region CN --interval 1d

        # normalize 1min cn
        python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/cn_data_1min --normalize_dir ~/.qlib/stock_data/source/cn_1min_nor --region CN --interval 1min

        # normalize 1d br
        python scripts/data_collector/yahoo/collector.py normalize_data --source_dir ~/.qlib/stock_data/source/br_data --normalize_dir ~/.qlib/stock_data/source/br_1d_nor --region BR --interval 1d

        # normalize 1min br
        python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/br_data --source_dir ~/.qlib/stock_data/source/br_data_1min --normalize_dir ~/.qlib/stock_data/source/br_1min_nor --region BR --interval 1min
        ```
  3. 生成二进制文件：`python scripts/dump_bin.py dump_all`
    
此步骤将把 `feature` 目录中标准化后的 csv 文件转换为 numpy 数组，并按列和股票代码分别存储归一化数据（每列一个文件，每个股票代码一个目录）。 
    
     - parameters:
       - `csv_path`: stock data path or directory, **normalize result(normalize_dir)**
       - `qlib_dir`: qlib(dump) data director
       - `freq`: transaction frequency, by default `day`
         > `freq_map = {1d:day, 1mih: 1min}`
       - `max_workers`: number of threads, by default *16*
       - `include_fields`: 需转储的字段，默认值为`""`
       - `exclude_fields`: 不需转储的字段，默认值为`""`
         > dump_fields = `include_fields if include_fields else set(symbol_df.columns) - set(exclude_fields) exclude_fields else symbol_df.columns`（如果指定include_fields则使用，否则使用所有列减去exclude_fields）
       - `symbol_field_name`: column *name* identifying symbol in csv files, by default `symbol`
       - `date_field_name`: column *name* identifying time in csv files, by default `date`
     - examples:
       ```bash
       # dump 1d cn
       python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/cn_data --freq day --exclude_fields date,symbol
       # dump 1min cn
       python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1min_nor --qlib_dir ~/.qlib/qlib_data/cn_data_1min --freq 1min --exclude_fields date,symbol
       ```

### Automatic update of daily frequency data(from yahoo finance)
  > 建议用户先手动更新一次数据（--trading_date 2021-05-25），然后再设置为自动更新。
  >
  > **注意**：用户无法基于 Qlib 提供的离线数据进行增量更新（为减小数据体积，部分字段已被移除）。用户应使用 [yahoo collector](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance) 从头开始下载 Yahoo 数据，然后进行增量更新。
  > 

   * 每个交易日自动更新数据到 "qlib" 目录（Linux）
      * 使用 *crontab*：`crontab -e`
      * 设置定时任务：

        ```
        * * * * 1-5 python <script path> update_data_to_bin --qlib_data_1d_dir <user data dir>
        ```
        * **script path**: *scripts/data_collector/yahoo/collector.py*

  * 手动更新数据
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --end_date <end date>
      ```
      * `end_date`: end of trading day(not included)
      * `check_data_length`: check the number of rows per *symbol*, by default `None`
        > if `len(symbol_df) < check_data_length`, it will be re-fetched, with the number of re-fetches coming from the `max_collector_count` parameter

  * `scripts/data_collector/yahoo/collector.py update_data_to_bin` 参数说明：
      * `source_dir`: The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
      * `normalize_dir`: Directory for normalize data, default "Path(__file__).parent/normalize"
      * `qlib_data_1d_dir`: the qlib data to be updated for yahoo, usually from: [download qlib data](https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)
      * `end_date`: end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
      * `region`: region, value from ["CN", "US"], default "CN"
      * `interval`: interval, default "1d"(Currently only supports 1d data)
      * `exists_skip`: exists skip, by default False

## Using qlib data

  ```python
  import qlib
  from qlib.data import D

  # 1d data cn
  # freq=day, freq default day
  qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
  df = D.features(D.instruments("all"), ["$close"], freq="day")

  # 1min data cn
  # freq=1min
  qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_1min", region="cn")
  inst = D.list_instruments(D.instruments("all"), freq="1min", as_list=True)
  # get 100 symbols
  df = D.features(inst[:100], ["$close"], freq="1min")
  # get all symbol data
  # df = D.features(D.instruments("all"), ["$close"], freq="1min")

  # 1d data us
  qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
  df = D.features(D.instruments("all"), ["$close"], freq="day")

  # 1min data us
  qlib.init(provider_uri="~/.qlib/qlib_data/us_data_1min", region="cn")
  inst = D.list_instruments(D.instruments("all"), freq="1min", as_list=True)
  # get 100 symbols
  df = D.features(inst[:100], ["$close"], freq="1min")
  # get all symbol data
  # df = D.features(D.instruments("all"), ["$close"], freq="1min")
  ```

