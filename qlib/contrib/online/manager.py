# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import pathlib
import pandas as pd
import shutil
from ruamel.yaml import YAML
from ...backtest.account import Account
from .user import User
from .utils import load_instance, save_instance
from ...utils import init_instance_by_config


class UserManager:
    def __init__(self, user_data_path, save_report=True):
        """
        该模块用于管理在线系统中的用户
        所有用户数据应保存在user_data_path中
            参数
                user_data_path : string
                    保存所有用户数据的路径

        变量:
            data_path : string
                保存所有用户数据的路径
            users_file : string
                记录用户添加日期的文件路径
            save_report : bool
                是否在每次交易过程后保存报告
            users : dict{}
                [user_id]->User()
                存储每个用户ID对应的User实例的Python字典
            user_record : pd.Dataframe
                user_id(string), add_date(string)
                指示每个用户的添加日期
        """
        self.data_path = pathlib.Path(user_data_path)
        self.users_file = self.data_path / "users.csv"
        self.save_report = save_report
        self.users = {}
        self.user_record = None

    def load_users(self):
        """
        将所有用户数据加载到管理器中
        """
        self.users = {}
        self.user_record = pd.read_csv(self.users_file, index_col=0)
        for user_id in self.user_record.index:
            self.users[user_id] = self.load_user(user_id)

    def load_user(self, user_id):
        """
        返回表示待处理用户的User()实例
            参数
                user_id : string
                用户ID
            :return
                user : User()
                用户实例
        """
        account_path = self.data_path / user_id
        strategy_file = self.data_path / user_id / "strategy_{}.pickle".format(user_id)
        model_file = self.data_path / user_id / "model_{}.pickle".format(user_id)
        cur_user_list = list(self.users)
        if user_id in cur_user_list:
            raise ValueError("User {} has been loaded".format(user_id))
        else:
            trade_account = Account(0)
            trade_account.load_account(account_path)
            strategy = load_instance(strategy_file)
            model = load_instance(model_file)
            user = User(account=trade_account, strategy=strategy, model=model)
            return user

    def save_user_data(self, user_id):
        """
        将User()实例保存到用户数据路径
            参数
                user_id : string
                用户ID
        """
        if not user_id in self.users:
            raise ValueError("Cannot find user {}".format(user_id))
        self.users[user_id].account.save_account(self.data_path / user_id)
        save_instance(
            self.users[user_id].strategy,
            self.data_path / user_id / "strategy_{}.pickle".format(user_id),
        )
        save_instance(
            self.users[user_id].model,
            self.data_path / user_id / "model_{}.pickle".format(user_id),
        )

    def add_user(self, user_id, config_file, add_date):
        """
        将新用户{user_id}添加到用户数据中
        将在用户数据路径中创建名为"{user_id}"的新文件夹
            参数
                user_id : string
                init_cash : int
                config_file : str/pathlib.Path()
                   配置文件路径
        """
        config_file = pathlib.Path(config_file)
        if not config_file.exists():
            raise ValueError("Cannot find config file {}".format(config_file))
        user_path = self.data_path / user_id
        if user_path.exists():
            raise ValueError("User data for {} already exists".format(user_id))

        with config_file.open("r") as fp:
            yaml = YAML(typ="safe", pure=True)
            config = yaml.load(fp)
        # load model
        model = init_instance_by_config(config["model"])

        # load strategy
        strategy = init_instance_by_config(config["strategy"])
        init_args = strategy.get_init_args_from_model(model, add_date)
        strategy.init(**init_args)

        # init Account
        trade_account = Account(init_cash=config["init_cash"])

        # save user
        user_path.mkdir()
        save_instance(model, self.data_path / user_id / "model_{}.pickle".format(user_id))
        save_instance(strategy, self.data_path / user_id / "strategy_{}.pickle".format(user_id))
        trade_account.save_account(self.data_path / user_id)
        user_record = pd.read_csv(self.users_file, index_col=0)
        user_record.loc[user_id] = [add_date]
        user_record.to_csv(self.users_file)

    def remove_user(self, user_id):
        """
        从当前用户数据集中移除用户{user_id}
        将删除用户数据路径中名为"{user_id}"的文件夹
            参数
                user_id : string
                用户ID
        """
        user_path = self.data_path / user_id
        if not user_path.exists():
            raise ValueError("Cannot find user data {}".format(user_id))
        shutil.rmtree(user_path)
        user_record = pd.read_csv(self.users_file, index_col=0)
        user_record.drop([user_id], inplace=True)
        user_record.to_csv(self.users_file)
