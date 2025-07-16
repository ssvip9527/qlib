# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import pathlib
import pickle
import pandas as pd
from ruamel.yaml import YAML
from ...data import D
from ...config import C
from ...log import get_module_logger
from ...utils import get_next_trading_date
from ...backtest.exchange import Exchange

log = get_module_logger("utils")


def load_instance(file_path):
    """
    加载pickle文件
        参数
           file_path : string / pathlib.Path()
                要加载的文件路径
        返回
            从文件加载的实例
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise ValueError("Cannot find file {}".format(file_path))
    with file_path.open("rb") as fr:
        instance = pickle.load(fr)
    return instance


def save_instance(instance, file_path):
    """
    将实例保存(序列化)到pickle文件
        参数
            instance :
                要序列化的数据
            file_path : string / pathlib.Path()
                要保存的文件路径
    """
    file_path = pathlib.Path(file_path)
    with file_path.open("wb") as fr:
        pickle.dump(instance, fr, C.dump_protocol_version)


def create_user_folder(path):
    """
    创建用户文件夹（如果不存在）
        参数
            path : string / pathlib.Path()
                要创建的文件夹路径
    """
    path = pathlib.Path(path)
    if path.exists():
        return
    path.mkdir(parents=True)
    head = pd.DataFrame(columns=("user_id", "add_date"))
    head.to_csv(path / "users.csv", index=None)


def prepare(um, today, user_id, exchange_config=None):
    """
    1. 获取用户{user_id}截至今天需要进行交易的日期
        dates[0]表示User{user_id}的最新交易日期，
        如果User{user_id}之前未进行过交易，则dates[0]表示User{user_id}的初始日期。
    2. 使用exchange_config文件设置交易所

        参数
            um : UserManager()
                用户管理器实例
            today : pd.Timestamp()
                今天的日期
            user_id : str
                用户ID
            exchange_config : str, optional
                交易所配置文件路径
        返回
            dates : list of pd.Timestamp
                交易日期列表
            trade_exchange : Exchange()
                交易所实例
    """
    # get latest trading date for {user_id}
    # if is None, indicate it haven't traded, then last trading date is init date of {user_id}
    latest_trading_date = um.users[user_id].get_latest_trading_date()
    if not latest_trading_date:
        latest_trading_date = um.user_record.loc[user_id][0]

    if str(today.date()) < latest_trading_date:
        log.warning("user_id:{}, last trading date {} after today {}".format(user_id, latest_trading_date, today))
        return [pd.Timestamp(latest_trading_date)], None

    dates = D.calendar(
        start_time=pd.Timestamp(latest_trading_date),
        end_time=pd.Timestamp(today),
        future=True,
    )
    dates = list(dates)
    dates.append(get_next_trading_date(dates[-1], future=True))
    if exchange_config:
        with pathlib.Path(exchange_config).open("r") as fp:
            yaml = YAML(typ="safe", pure=True)
            exchange_paras = yaml.load(fp)
    else:
        exchange_paras = {}
    trade_exchange = Exchange(trade_dates=dates, **exchange_paras)
    return dates, trade_exchange
