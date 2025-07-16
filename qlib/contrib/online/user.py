# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import logging

from ...log import get_module_logger
from ..evaluate import risk_analysis
from ...data import D


class User:
    def __init__(self, account, strategy, model, verbose=False):
        """
        在线系统中的用户，包含账户、策略和模型三个模块。
            参数
                account : Account()
                    账户实例
                strategy :
                    策略实例
                model :
                    a model instance
                report_save_path : string
                    the path to save report. Will not save report if None
                verbose : bool
                    是否在过程中打印信息
        """
        self.logger = get_module_logger("User", level=logging.INFO)
        self.account = account
        self.strategy = strategy
        self.model = model
        self.verbose = verbose

    def init_state(self, date):
        """
        每个交易日开始时初始化状态
            参数
                date : pd.Timestamp
                    日期
        """
        self.account.init_state(today=date)
        self.strategy.init_state(trade_date=date, model=self.model, account=self.account)
        return

    def get_latest_trading_date(self):
        """
        返回用户{user_id}的最新交易日期
            参数
                user_id : string
                用户ID
            返回
                date : string (例如 '2018-10-08')
                最新交易日期字符串
        """
        if not self.account.last_trade_date:
            return None
        return str(self.account.last_trade_date.date())

    def showReport(self, benchmark="SH000905"):
        """
        显示最新报告(均值、标准差、信息比率、年化收益率)
            参数
                benchmark : string
                    用于比较的基准，'SH000905'代表沪深500
        """
        bench = D.features([benchmark], ["$change"], disk_cache=True).loc[benchmark, "$change"]
        portfolio_metrics = self.account.portfolio_metrics.generate_portfolio_metrics_dataframe()
        portfolio_metrics["bench"] = bench
        analysis_result = {"pred": {}, "excess_return_without_cost": {}, "excess_return_with_cost": {}}
        r = (portfolio_metrics["return"] - portfolio_metrics["bench"]).dropna()
        analysis_result["excess_return_without_cost"][0] = risk_analysis(r)
        r = (portfolio_metrics["return"] - portfolio_metrics["bench"] - portfolio_metrics["cost"]).dropna()
        analysis_result["excess_return_with_cost"][0] = risk_analysis(r)
        self.logger.info("Result of porfolio:")
        self.logger.info("excess_return_without_cost:")
        self.logger.info(analysis_result["excess_return_without_cost"][0])
        self.logger.info("excess_return_with_cost:")
        self.logger.info(analysis_result["excess_return_with_cost"][0])
        return portfolio_metrics
