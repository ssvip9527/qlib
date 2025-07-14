## 数据收集

### 获取Qlib数据(`bin文件`)

  - get data: `python scripts/get_data.py qlib_data`
  - parameters:
    - `target_dir`: 保存目录，默认值为*~/.qlib/qlib_data/cn_data_5min*
    - `version`: 数据集版本，取值为[`v2`]，默认值为`v2`
      - `v2`的结束日期为*2022-12*
    - `interval`: 数据频率，固定为`5min`
    - `region`: 市场区域，固定为`hs300`
    - `delete_old`: 删除`target_dir`中已存在的数据(*features, calendars, instruments, dataset_cache, features_cache*)，取值为[`True`, `False`]，默认值为`True`
    - `exists_skip`: 若目标目录数据已存在则跳过`get_data`，取值为[`True`, `False`]，默认值为`False`
  - examples:
    ```bash
    # hs300 5min
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/hs300_data_5min --region hs300 --interval 5min
    ```
    
### 收集*宝钢高频*数据到qlib
> 收集*宝钢高频*数据并*转储*为`qlib`格式。
> 如果上述现成数据无法满足用户需求，用户可以按照本部分内容爬取最新数据并转换为qlib数据格式。
  1. 下载数据到csv：`python scripts/data_collector/baostock_5min/collector.py download_data`
     
     这将从宝钢下载日期、股票代码、开盘价、最高价、最低价、收盘价、成交量、成交额、复权标志等原始数据到本地目录，每个股票代码对应一个文件。
     - 参数说明：
          - `source_dir`: 保存目录
          - `interval`: 数据频率，固定为`5min`
          - `region`: 市场区域，固定为`HS300`
          - `start`: 开始时间，默认值为*None*
          - `end`: 结束时间，默认值为*None*
     - examples:
          ```bash
          # cn 5min data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
          ```
  2. 数据标准化：`python scripts/data_collector/baostock_5min/collector.py normalize_data`
     
     此步骤将：
     1. 使用复权收盘价对最高价、最低价、收盘价、开盘价进行标准化处理。
     2. 对最高价、最低价、收盘价、开盘价进行归一化，使第一个有效交易日的收盘价为 1。 
     - 参数说明：
          - `source_dir`: csv文件目录
          - `normalize_dir`: 结果保存目录
          - `interval`: 数据频率，固定为`5min`
            > 若**`interval == 5min`**，则`qlib_data_1d_dir`不能为空
          - `region`: 市场区域，固定为`HS300`
          - `date_field_name`: csv文件中标识时间的列名，默认值为`date`
          - `symbol_field_name`: csv文件中标识股票代码的列名，默认值为`symbol`
          - `end_date`: 若不为`None`，则标准化保存的最后日期（包含end_date）；若为`None`，则忽略此参数；默认值为`None`
          - `qlib_data_1d_dir`: qlib格式的1日数据目录
            if interval==5min, qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;
            ```
                # qlib_data_1d can be obtained like this:
                python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            ```
      - examples:
        ```bash
        # normalize 5min cn
        python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
        ```
  3. dump data: `python scripts/dump_bin.py dump_all`
    
     此步骤将把`feature`目录中标准化后的csv文件转换为numpy数组，并按列和股票代码分别存储标准化数据。 
    
     - parameters:
       - `csv_path`: stock data path or directory, **normalize result(normalize_dir)**
       - `qlib_dir`: qlib(dump) data director
       - `freq`: transaction frequency, by default `day`
         > `freq_map = {1d:day, 5mih: 5min}`
       - `max_workers`: 线程数量，默认值为*16*
       - `include_fields`: 需转储的字段，默认值为`""`
       - `exclude_fields`: 不需转储的字段，默认值为`""`
         > dump_fields = `include_fields if include_fields else set(symbol_df.columns) - set(exclude_fields) exclude_fields else symbol_df.columns`（如果指定include_fields则使用，否则使用所有列减去exclude_fields）
       - `symbol_field_name`: csv文件中标识股票代码的列名，默认值为`symbol`
       - `date_field_name`: csv文件中标识时间的列名，默认值为`date`
     - examples:
       ```bash
       # 生成5分钟数据二进制文件（中国市场）
       python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/hs300_5min_nor --qlib_dir ~/.qlib/qlib_data/hs300_5min_bin --freq 5min --exclude_fields date,symbol
       ```