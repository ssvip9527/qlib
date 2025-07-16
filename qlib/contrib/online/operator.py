# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import fire
import pandas as pd
import pathlib
import qlib
import logging

from ...data import D
from ...log import get_module_logger
from ...utils import get_pre_trading_date, is_tradable_date
from ..evaluate import risk_analysis
from ..backtest.backtest import update_account

from .manager import UserManager
from .utils import prepare
from .utils import create_user_folder
from .executor import load_order_list, save_order_list
from .executor import SimulatorExecutor
from .executor import save_score_series, load_score_series


class Operator:
    def __init__(self, client: str):
        """
        参数
        ----------
            client: str
                qlib客户端配置文件(.yaml)
        """
        self.logger = get_module_logger("online operator", level=logging.INFO)
        self.client = client

    @staticmethod
    def init(client, path, date=None):
        """初始化UserManager()，获取预测日期和交易日期
        参数
        ----------
            client: str
                qlib客户端配置文件(.yaml)
            path : str
                保存用户账户的路径。
            date : str (YYYY-MM-DD)
                交易日期，生成的订单列表将在该日期交易。
        返回
        ----------
            um: UserManager()
                用户管理器实例
            pred_date: pd.Timestamp
                预测日期
            trade_date: pd.Timestamp
                交易日期
        """
        qlib.init_from_yaml_conf(client)
        um = UserManager(user_data_path=pathlib.Path(path))
        um.load_users()
        if not date:
            trade_date, pred_date = None, None
        else:
            trade_date = pd.Timestamp(date)
            if not is_tradable_date(trade_date):
                raise ValueError("trade date is not tradable date".format(trade_date.date()))
            pred_date = get_pre_trading_date(trade_date, future=True)
        return um, pred_date, trade_date

    def add_user(self, id, config, path, date):
        """将新用户添加到文件夹以运行'online'模块。

        参数
        ----------
        id : str
            用户ID，应唯一。
        config : str
            用户配置文件路径(yaml)
        path : str
            保存用户账户的路径。
        date : str (YYYY-MM-DD)
            用户账户添加的日期。
        """
        create_user_folder(path)
        qlib.init_from_yaml_conf(self.client)
        um = UserManager(user_data_path=path)
        add_date = D.calendar(end_time=date)[-1]
        if not is_tradable_date(add_date):
            raise ValueError("add date is not tradable date".format(add_date.date()))
        um.add_user(user_id=id, config_file=config, add_date=add_date)

    def remove_user(self, id, path):
        """从'online'模块使用的文件夹中移除用户。

        参数
        ----------
        id : str
            用户ID，应唯一。
        path : str
            保存用户账户的路径。
        """
        um = UserManager(user_data_path=path)
        um.remove_user(user_id=id)

    def generate(self, date, path):
        """生成将在'date'进行交易的订单列表。

        参数
        ----------
        date : str (YYYY-MM-DD)
            交易日期，生成的订单列表将在该日期交易。
        path : str
            保存用户账户的路径。
        """
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, pred_date, user_id)
            # get and save the score at predict date
            input_data = user.model.get_data_with_date(pred_date)
            score_series = user.model.predict(input_data)
            save_score_series(score_series, (pathlib.Path(path) / user_id), trade_date)

            # update strategy (and model)
            user.strategy.update(score_series, pred_date, trade_date)

            # generate and save order list
            order_list = user.strategy.generate_trade_decision(
                score_series=score_series,
                current=user.account.current_position,
                trade_exchange=trade_exchange,
                trade_date=trade_date,
            )
            save_order_list(
                order_list=order_list,
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            self.logger.info("Generate order list at {} for {}".format(trade_date, user_id))
            um.save_user_data(user_id)

    def execute(self, date, exchange_config, path):
        """在'date'执行订单列表。

        参数
        ----------
           date : str (YYYY-MM-DD)
               交易日期，生成的订单列表将在该日期交易。
           exchange_config: str
               交易所配置文件路径(yaml)
           path : str
               保存用户账户的路径。
        """
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, trade_date, user_id, exchange_config)
            executor = SimulatorExecutor(trade_exchange=trade_exchange)
            if str(dates[0].date()) != str(pred_date.date()):
                raise ValueError(
                    "The account data is not newest! last trading date {}, today {}".format(
                        dates[0].date(), trade_date.date()
                    )
                )

            # load and execute the order list
            # will not modify the trade_account after executing
            order_list = load_order_list(user_path=(pathlib.Path(path) / user_id), trade_date=trade_date)
            trade_info = executor.execute(order_list=order_list, trade_account=user.account, trade_date=trade_date)
            executor.save_executed_file_from_trade_info(
                trade_info=trade_info,
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            self.logger.info("execute order list at {} for {}".format(trade_date.date(), user_id))

    def update(self, date, path, type="SIM"):
        """在'date'更新账户。

        参数
        ----------
        date : str (YYYY-MM-DD)
            交易日期，生成的订单列表将在该日期交易。
        path : str
            保存用户账户的路径。
        type : str
            用于执行订单列表的执行器类型
            'SIM': 模拟执行器(SimulatorExecutor)
        """
        if type not in ["SIM", "YC"]:
            raise ValueError("type is invalid, {}".format(type))
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, trade_date, user_id)
            if type == "SIM":
                executor = SimulatorExecutor(trade_exchange=trade_exchange)
            else:
                raise ValueError("not found executor")
            # dates[0] is the last_trading_date
            if str(dates[0].date()) > str(pred_date.date()):
                raise ValueError(
                    "The account data is not newest! last trading date {}, today {}".format(
                        dates[0].date(), trade_date.date()
                    )
                )
            # load trade info and update account
            trade_info = executor.load_trade_info_from_executed_file(
                user_path=(pathlib.Path(path) / user_id), trade_date=trade_date
            )
            score_series = load_score_series((pathlib.Path(path) / user_id), trade_date)
            update_account(user.account, trade_info, trade_exchange, trade_date)

            portfolio_metrics = user.account.portfolio_metrics.generate_portfolio_metrics_dataframe()
            self.logger.info(portfolio_metrics)
            um.save_user_data(user_id)
            self.logger.info("Update account state {} for {}".format(trade_date, user_id))

    def simulate(self, id, config, exchange_config, start, end, path, bench="SH000905"):
        """从开始日期到结束日期，每天运行(生成交易决策->执行订单列表->更新账户)流程。

        参数
        ----------
        id : str
            用户ID，需唯一
        config : str
            用户配置文件路径(yaml)
        exchange_config: str
            交易所配置文件路径(yaml)
        start : str "YYYY-MM-DD"
            运行在线模拟的开始日期
        end : str "YYYY-MM-DD"
            运行在线模拟的结束日期
        path : str
            保存用户账户的路径。
        bench : str
            用于比较结果的基准。
            'SH000905'代表沪深500，'SH000300'代表沪深300
        """
        # Clear the current user if exists, then add a new user.
        create_user_folder(path)
        um = self.init(self.client, path, None)[0]
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
        try:
            um.remove_user(user_id=id)
        except BaseException:
            pass
        um.add_user(user_id=id, config_file=config, add_date=pd.Timestamp(start_date))

        # Do the online simulate
        um.load_users()
        user = um.users[id]
        dates, trade_exchange = prepare(um, end_date, id, exchange_config)
        executor = SimulatorExecutor(trade_exchange=trade_exchange)
        for pred_date, trade_date in zip(dates[:-2], dates[1:-1]):
            user_path = pathlib.Path(path) / id

            # 1. load and save score_series
            input_data = user.model.get_data_with_date(pred_date)
            score_series = user.model.predict(input_data)
            save_score_series(score_series, (pathlib.Path(path) / id), trade_date)

            # 2. update strategy (and model)
            user.strategy.update(score_series, pred_date, trade_date)

            # 3. generate and save order list
            order_list = user.strategy.generate_trade_decision(
                score_series=score_series,
                current=user.account.current_position,
                trade_exchange=trade_exchange,
                trade_date=trade_date,
            )
            save_order_list(order_list=order_list, user_path=user_path, trade_date=trade_date)

            # 4. auto execute order list
            order_list = load_order_list(user_path=user_path, trade_date=trade_date)
            trade_info = executor.execute(trade_account=user.account, order_list=order_list, trade_date=trade_date)
            executor.save_executed_file_from_trade_info(
                trade_info=trade_info, user_path=user_path, trade_date=trade_date
            )
            # 5. update account state
            trade_info = executor.load_trade_info_from_executed_file(user_path=user_path, trade_date=trade_date)
            update_account(user.account, trade_info, trade_exchange, trade_date)
        portfolio_metrics = user.account.portfolio_metrics.generate_portfolio_metrics_dataframe()
        self.logger.info(portfolio_metrics)
        um.save_user_data(id)
        self.show(id, path, bench)

    def show(self, id, path, bench="SH000905"):
        """显示最新报告(均值、标准差、信息比率、年化收益率)

        参数
        ----------
        id : str
            用户ID，需唯一
        path : str
            保存用户账户的路径。
        bench : str
            用于比较结果的基准。
            'SH000905'代表沪深500，'SH000300'代表沪深300
        """
        um = self.init(self.client, path, None)[0]
        if id not in um.users:
            raise ValueError("Cannot find user ".format(id))
        bench = D.features([bench], ["$change"]).loc[bench, "$change"]
        portfolio_metrics = um.users[id].account.portfolio_metrics.generate_portfolio_metrics_dataframe()
        portfolio_metrics["bench"] = bench
        analysis_result = {}
        r = (portfolio_metrics["return"] - portfolio_metrics["bench"]).dropna()
        analysis_result["excess_return_without_cost"] = risk_analysis(r)
        r = (portfolio_metrics["return"] - portfolio_metrics["bench"] - portfolio_metrics["cost"]).dropna()
        analysis_result["excess_return_with_cost"] = risk_analysis(r)
        print("Result:")
        print("excess_return_without_cost:")
        print(analysis_result["excess_return_without_cost"])
        print("excess_return_with_cost:")
        print(analysis_result["excess_return_with_cost"])


def run():
    fire.Fire(Operator)


if __name__ == "__main__":
    run()
