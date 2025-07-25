# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 我们使用ArcticDB替代Arctic，因为：
# - ArcticDB是Arctic的继任者，于2023年3月发布
# - ArcticDB提供更好的性能和更简单的安装方式
# - ArcticDB不需要额外的服务器，可以使用本地存储或S3

import arcticdb as adb
import pandas as pd

from qlib.data.data import FeatureProvider


class ArcticDBFeatureProvider(FeatureProvider):
    def __init__(
        self, uri="lmdb:///tmp/arcticdb", retry_time=0, market_transaction_time_list=None
    ):
        """
        初始化ArcticDBFeatureProvider
        
        参数:
            uri: str
                ArcticDB连接URI，可以是lmdb本地路径或S3路径
                例如："lmdb:///tmp/arcticdb" 或 "s3://ENDPOINT:BUCKET?aws_auth=true"
            retry_time: int
                发生错误时重试连接的次数
            market_transaction_time_list: list
                市场交易时间列表，默认为[('09:15', '11:30'), ('13:00', '15:00')]
        """
        super().__init__()
        if market_transaction_time_list is None:
            market_transaction_time_list = [("09:15", "11:30"), ("13:00", "15:00")]
        self.uri = uri
        self.retry_time = retry_time
        # 注意：这对于TResample算子尤为重要
        self.market_transaction_time_list = market_transaction_time_list

    def feature(self, instrument, field, start_index, end_index, freq):
        """
        获取特定字段的特征数据
        
        参数:
            instrument: str
                股票代码
            field: str
                字段名称
            start_index: datetime
                开始时间
            end_index: datetime
                结束时间
            freq: str
                频率，如"ticks"、"transaction"、"order"等
                
        返回:
            pd.Series: 特征数据
        """
        field = str(field)[1:]
        # 创建ArcticDB连接
        arctic = adb.Arctic(self.uri)
        
        # 检查库是否存在
        if freq not in arctic.list_libraries():
            raise ValueError("lib {} not in arcticdb".format(freq))
        
        # 获取库
        lib = arctic[freq]
        
        # 检查股票代码是否存在
        if instrument not in lib.list_symbols():
            # 股票代码不存在
            return pd.Series()
        else:
            # 读取数据
            # ArcticDB的read方法与Arctic略有不同
            # 使用date_range参数代替chunk_range
            df = lib.read(instrument, columns=[field], date_range=(start_index, end_index))
            s = df[field]

            if not s.empty:
                s = pd.concat(
                    [
                        s.between_time(time_tuple[0], time_tuple[1])
                        for time_tuple in self.market_transaction_time_list
                    ]
                )
            return s