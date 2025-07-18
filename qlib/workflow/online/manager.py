# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineManager可以管理一组`Online Strategy <#Online Strategy>`_并动态运行它们。

随着时间的推移，决策模型也会发生变化。在本模块中，我们将这些贡献模型称为`online`模型。
在每个例行程序(如每天或每分钟)中，`online`模型可能会发生变化，需要更新它们的预测。
因此本模块提供了一系列方法来控制这个过程。

本模块还提供了一种在历史中模拟`Online Strategy <#Online Strategy>`_的方法。
这意味着您可以验证您的策略或找到更好的策略。

在不同情况下使用不同训练器共有4种情况:



=========================  ===================================================================================
情况                      描述
=========================  ===================================================================================
在线+训练器                当您想要执行真实的例行程序时，训练器将帮助您训练模型。
                           它会逐个任务、逐个策略地训练模型。

在线+延迟训练器            延迟训练器将跳过具体训练，直到所有任务已由不同策略准备完毕。
                           它使用户可以在`routine`或`first_train`结束时并行训练所有任务。
                           否则，当每个策略准备任务时，这些函数会卡住。

模拟+训练器                它的行为方式与`在线+训练器`相同。唯一的区别是它用于模拟/回测而非在线交易

模拟+延迟训练器            当您的模型没有任何时间依赖性时，您可以使用延迟训练器来实现多任务处理能力。
                           这意味着所有例行程序中的所有任务都可以在模拟结束时真实训练。
                           信号将在不同时间段准备好(基于是否有任何新模型在线)。
=========================  ===================================================================================

以下是一些伪代码，展示了每种情况的工作流程

为简单起见
    - 策略中只使用一个策略
    - `update_online_pred`仅在在线模式下调用并被忽略

1) `在线+训练器`

.. code-block:: python

    tasks = first_train()
    models = trainer.train(tasks)
    trainer.end_train(models)
    for day in online_trading_days:
        # OnlineManager.routine
        models = trainer.train(strategy.prepare_tasks())  # 对每个策略
        strategy.prepare_online_models(models)  # 对每个策略

        trainer.end_train(models)
        prepare_signals()  # 每日准备交易信号


`在线+延迟训练器`: 工作流程与`在线+训练器`相同。


2) `模拟+延迟训练器`

.. code-block:: python

    # 模拟
    tasks = first_train()
    models = trainer.train(tasks)
    for day in historical_calendars:
        # OnlineManager.routine
        models = trainer.train(strategy.prepare_tasks())  # 对每个策略
        strategy.prepare_online_models(models)  # 对每个策略
    # delay_prepare()
    # FIXME: 目前delay_prepare没有以正确的方式实现。
    trainer.end_train(<for all previous models>)
    prepare_signals()


# 我们可以简化当前的工作流程吗?

- 可以减少任务的状态数量吗?

    - 对于每个任务，我们有三个阶段(即任务、部分训练的任务、最终训练的任务)
"""

import logging
from typing import Callable, List, Union

import pandas as pd
from qlib import get_module_logger
from qlib.data.data import D
from qlib.log import set_global_logger_level
from qlib.model.ens.ensemble import AverageEnsemble
from qlib.model.trainer import Trainer, TrainerR
from qlib.utils.serial import Serializable
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.task.collect import MergeCollector


class OnlineManager(Serializable):
    """
    OnlineManager可以通过`Online Strategy <#Online Strategy>`_管理在线模型。
    它还提供了哪些模型在什么时间在线的历史记录。
    """

    STATUS_SIMULATING = "simulating"  # when calling `simulate`
    STATUS_ONLINE = "online"  # the normal status. It is used when online trading

    def __init__(
        self,
        strategies: Union[OnlineStrategy, List[OnlineStrategy]],
        trainer: Trainer = None,
        begin_time: Union[str, pd.Timestamp] = None,
        freq="day",
    ):
        """
        初始化OnlineManager。
        一个OnlineManager必须至少有一个OnlineStrategy。

        参数:
            strategies (Union[OnlineStrategy, List[OnlineStrategy]]): OnlineStrategy实例或OnlineStrategy列表
            begin_time (Union[str,pd.Timestamp], 可选): OnlineManager将在此时间开始。默认为None表示使用最新日期。
            trainer (qlib.model.trainer.Trainer): 用于训练任务的训练器。None表示使用TrainerR。
            freq (str, 可选): 数据频率。默认为"day"。
        """
        self.logger = get_module_logger(self.__class__.__name__)
        if not isinstance(strategies, list):
            strategies = [strategies]
        self.strategies = strategies
        self.freq = freq
        if begin_time is None:
            begin_time = D.calendar(freq=self.freq).max()
        self.begin_time = pd.Timestamp(begin_time)
        self.cur_time = self.begin_time
        # OnlineManager will recorder the history of online models, which is a dict like {pd.Timestamp, {strategy, [online_models]}}.
        # It records the online servnig models of each strategy for each day.
        self.history = {}
        if trainer is None:
            trainer = TrainerR()
        self.trainer = trainer
        self.signals = None
        self.status = self.STATUS_ONLINE

    def _postpone_action(self):
        """
        是否将以下操作推迟到最后(在delay_prepare中)
        - trainer.end_train
        - prepare_signals

        推迟这些操作是为了支持没有时间依赖性的模拟/回测在线策略。
        所有操作都可以在最后并行完成。
        """
        return self.status == self.STATUS_SIMULATING and self.trainer.is_delay()

    def first_train(self, strategies: List[OnlineStrategy] = None, model_kwargs: dict = {}):
        """
        从每个策略的first_tasks方法获取任务并训练它们。
        如果使用DelayTrainer，它可以在每个策略的first_tasks之后一起完成所有训练。

        参数:
            strategies (List[OnlineStrategy]): 策略列表(添加策略时需要此参数)。None表示使用默认策略。
            model_kwargs (dict): `prepare_online_models`的参数
        """
        if strategies is None:
            strategies = self.strategies

        models_list = []
        for strategy in strategies:
            self.logger.info(f"Strategy `{strategy.name_id}` begins first training...")
            tasks = strategy.first_tasks()
            models = self.trainer.train(tasks, experiment_name=strategy.name_id)
            models_list.append(models)
            self.logger.info(f"Finished training {len(models)} models.")
            # FIXME: Train multiple online models at `first_train` will result in getting too much online models at the
            # start.
            online_models = strategy.prepare_online_models(models, **model_kwargs)
            self.history.setdefault(self.cur_time, {})[strategy] = online_models

        if not self._postpone_action():
            for strategy, models in zip(strategies, models_list):
                models = self.trainer.end_train(models, experiment_name=strategy.name_id)

    def routine(
        self,
        cur_time: Union[str, pd.Timestamp] = None,
        task_kwargs: dict = {},
        model_kwargs: dict = {},
        signal_kwargs: dict = {},
    ):
        """
        每个策略的典型更新过程并记录在线历史。

        例行程序(如逐日或逐月)后的典型更新过程。
        过程是: 更新预测 -> 准备任务 -> 准备在线模型 -> 准备信号。

        如果使用DelayTrainer，它可以在每个策略的prepare_tasks之后一起完成所有训练。

        参数:
            cur_time (Union[str,pd.Timestamp], 可选): 在此时间运行routine方法。默认为None。
            task_kwargs (dict): `prepare_tasks`的参数
            model_kwargs (dict): `prepare_online_models`的参数
            signal_kwargs (dict): `prepare_signals`的参数
        """
        if cur_time is None:
            cur_time = D.calendar(freq=self.freq).max()
        self.cur_time = pd.Timestamp(cur_time)  # None for latest date

        models_list = []
        for strategy in self.strategies:
            self.logger.info(f"Strategy `{strategy.name_id}` begins routine...")

            tasks = strategy.prepare_tasks(self.cur_time, **task_kwargs)
            models = self.trainer.train(tasks, experiment_name=strategy.name_id)
            models_list.append(models)
            self.logger.info(f"Finished training {len(models)} models.")
            online_models = strategy.prepare_online_models(models, **model_kwargs)
            self.history.setdefault(self.cur_time, {})[strategy] = online_models

            # The online model may changes in the above processes
            # So updating the predictions of online models should be the last step
            if self.status == self.STATUS_ONLINE:
                strategy.tool.update_online_pred()

        if not self._postpone_action():
            for strategy, models in zip(self.strategies, models_list):
                models = self.trainer.end_train(models, experiment_name=strategy.name_id)
            self.prepare_signals(**signal_kwargs)

    def get_collector(self, **kwargs) -> MergeCollector:
        """
        获取`Collector <../advanced/task_management.html#Task Collecting>`_实例以收集每个策略的结果。
        此收集器可以作为信号准备的基础。

        参数:
            **kwargs: get_collector的参数。

        返回:
            MergeCollector: 用于合并其他收集器的收集器。
        """
        collector_dict = {}
        for strategy in self.strategies:
            collector_dict[strategy.name_id] = strategy.get_collector(**kwargs)
        return MergeCollector(collector_dict, process_list=[])

    def add_strategy(self, strategies: Union[OnlineStrategy, List[OnlineStrategy]]):
        """
        向OnlineManager添加一些新策略。

        参数:
            strategy (Union[OnlineStrategy, List[OnlineStrategy]]): OnlineStrategy列表
        """
        if not isinstance(strategies, list):
            strategies = [strategies]
        self.first_train(strategies)
        self.strategies.extend(strategies)

    def prepare_signals(self, prepare_func: Callable = AverageEnsemble(), over_write=False):
        """
        在准备完最后一个例行程序(箱线图中的一个框)的数据后，这意味着例行程序的结束，我们可以为下一个例行程序准备交易信号。

        注意: 给定一组预测，这些预测结束时间之前的所有信号都将准备好。

        即使最新的信号已经存在，最新的计算结果也将被覆盖。

        .. note::

            给定某个时间的预测，此时间之前的所有信号都将准备好。

        参数:
            prepare_func (Callable, 可选): 从收集后的字典中获取信号。默认为AverageEnsemble()，由MergeCollector收集的结果必须是{xxx:pred}。
            over_write (bool, 可选): 如果为True，新信号将覆盖。如果为False，新信号将附加到信号末尾。默认为False。

        返回:
            pd.DataFrame: 信号。
        """
        signals = prepare_func(self.get_collector()())
        old_signals = self.signals
        if old_signals is not None and not over_write:
            old_max = old_signals.index.get_level_values("datetime").max()
            new_signals = signals.loc[old_max:]
            signals = pd.concat([old_signals, new_signals], axis=0)
        else:
            new_signals = signals
        self.logger.info(f"Finished preparing new {len(new_signals)} signals.")
        self.signals = signals
        return new_signals

    def get_signals(self) -> Union[pd.Series, pd.DataFrame]:
        """
        获取准备好的在线信号。

        返回:
            Union[pd.Series, pd.DataFrame]: pd.Series表示每个日期时间只有一个信号。
            pd.DataFrame表示多个信号，例如买卖操作使用不同的交易信号。
        """
        return self.signals

    SIM_LOG_LEVEL = logging.INFO + 1  # when simulating, reduce information
    SIM_LOG_NAME = "SIMULATE_INFO"

    def simulate(
        self, end_time=None, frequency="day", task_kwargs={}, model_kwargs={}, signal_kwargs={}
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        从当前时间开始，此方法将模拟OnlineManager中的每个例行程序，直到结束时间。

        考虑到并行训练，模型和信号可以在所有例行程序模拟后准备。

        延迟训练方式可以是``DelayTrainer``，延迟准备信号方式可以是``delay_prepare``。

        参数:
            end_time: 模拟结束的时间
            frequency: 日历频率
            task_kwargs (dict): `prepare_tasks`的参数
            model_kwargs (dict): `prepare_online_models`的参数
            signal_kwargs (dict): `prepare_signals`的参数

        返回:
            Union[pd.Series, pd.DataFrame]: pd.Series表示每个日期时间只有一个信号。
            pd.DataFrame表示多个信号，例如买卖操作使用不同的交易信号。
        """
        self.status = self.STATUS_SIMULATING
        cal = D.calendar(start_time=self.cur_time, end_time=end_time, freq=frequency)
        self.first_train()

        simulate_level = self.SIM_LOG_LEVEL
        set_global_logger_level(simulate_level)
        logging.addLevelName(simulate_level, self.SIM_LOG_NAME)

        for cur_time in cal:
            self.logger.log(level=simulate_level, msg=f"Simulating at {str(cur_time)}......")
            self.routine(
                cur_time,
                task_kwargs=task_kwargs,
                model_kwargs=model_kwargs,
                signal_kwargs=signal_kwargs,
            )
        # delay prepare the models and signals
        if self._postpone_action():
            self.delay_prepare(model_kwargs=model_kwargs, signal_kwargs=signal_kwargs)

        # FIXME: get logging level firstly and restore it here
        set_global_logger_level(logging.DEBUG)
        self.logger.info(f"Finished preparing signals")
        self.status = self.STATUS_ONLINE
        return self.get_signals()

    def delay_prepare(self, model_kwargs={}, signal_kwargs={}):
        """
        如果有任何内容等待准备，则准备所有模型和信号。

        参数:
            model_kwargs: `end_train`的参数
            signal_kwargs: `prepare_signals`的参数
        """
        # FIXME:
        # This method is not implemented in the proper way!!!
        last_models = {}
        signals_time = D.calendar()[0]
        need_prepare = False
        for cur_time, strategy_models in self.history.items():
            self.cur_time = cur_time

            for strategy, models in strategy_models.items():
                # only new online models need to prepare
                if last_models.setdefault(strategy, set()) != set(models):
                    models = self.trainer.end_train(models, experiment_name=strategy.name_id, **model_kwargs)
                    strategy.tool.reset_online_tag(models)
                    need_prepare = True
                last_models[strategy] = set(models)

            if need_prepare:
                # NOTE: Assumption: the predictions of online models need less than next cur_time, or this method will work in a wrong way.
                self.prepare_signals(**signal_kwargs)
                if signals_time > cur_time:
                    # FIXME: if use DelayTrainer and worker (and worker is faster than main progress), there are some possibilities of showing this warning.
                    self.logger.warn(
                        f"The signals have already parpred to {signals_time} by last preparation, but current time is only {cur_time}. This may be because the online models predict more than they should, which can cause signals to be contaminated by the offline models."
                    )
                need_prepare = False
                signals_time = self.signals.index.get_level_values("datetime").max()
