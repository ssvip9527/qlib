# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 我们将arctic从Qlib的核心框架移至contrib目录，原因如下
# - Arctic对pandas和numpy版本有非常严格的限制
#    - https://github.com/man-group/arctic/pull/908
# - pip无法正确计算版本号！！！
#    - 也许我们可以通过poetry解决此问题

# FIXME: 因此，如果你想使用基于arctic的提供器，请手动安装arctic
# `pip install arctic` may not be enough.
from arctic import Arctic
import pandas as pd
import pymongo

from qlib.data.data import FeatureProvider


class ArcticFeatureProvider(FeatureProvider):
    def __init__(
        self, uri="127.0.0.1", retry_time=0, market_transaction_time_list=[("09:15", "11:30"), ("13:00", "15:00")]
    ):
        super().__init__()
        self.uri = uri
        # TODO:
        # 发生错误时重试连接
        # 这真的重要吗？
        self.retry_time = retry_time
        # 注意：这对于TResample算子尤为重要
        self.market_transaction_time_list = market_transaction_time_list

    def feature(self, instrument, field, start_index, end_index, freq):
        field = str(field)[1:]
        with pymongo.MongoClient(self.uri) as client:
            # TODO: 这会导致频繁连接服务器并引发性能问题
            arctic = Arctic(client)

            if freq not in arctic.list_libraries():
                raise ValueError("lib {} not in arctic".format(freq))

            if instrument not in arctic[freq].list_symbols():
                # instruments does not exist
                return pd.Series()
            else:
                df = arctic[freq].read(instrument, columns=[field], chunk_range=(start_index, end_index))
                s = df[field]

                if not s.empty:
                    s = pd.concat(
                        [
                            s.between_time(time_tuple[0], time_tuple[1])
                            for time_tuple in self.market_transaction_time_list
                        ]
                    )
                return s
