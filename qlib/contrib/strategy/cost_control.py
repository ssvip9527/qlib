# 版权所有 (c) Microsoft Corporation.
# 根据MIT许可证授权
"""
此策略未得到良好维护
"""


from .order_generator import OrderGenWInteract
from .signal_strategy import WeightStrategyBase
import copy


class SoftTopkStrategy(WeightStrategyBase):
    def __init__(
        self,
        model,
        dataset,
        topk,
        order_generator_cls_or_obj=OrderGenWInteract,
        max_sold_weight=1.0,
        risk_degree=0.95,
        buy_method="first_fill",
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        参数
        ----------
        topk : int
            要购买的前N只股票
        risk_degree : float
            总价值的持仓百分比。buy_method 选项：
                rank_fill: 优先为排名靠前的股票分配权重（最大1/topk）
                average_fill: 为排名靠前的股票平均分配权重。
        """
        super(SoftTopkStrategy, self).__init__(
            model, dataset, order_generator_cls_or_obj, trade_exchange, level_infra, common_infra, **kwargs
        )
        self.topk = topk
        self.max_sold_weight = max_sold_weight
        self.risk_degree = risk_degree
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        返回将用于投资的总价值比例。动态调整risk_degree将导致市场择时。
        """
        # 默认情况下，将使用您总价值的95%
        return self.risk_degree

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        参数
        ----------
        score:
            该交易日的预测分数，pd.Series类型，索引为股票ID，包含'score'列
        current:
            当前持仓，使用Position()类
        trade_date:
            交易日

            根据当日分数和当前持仓生成目标持仓。
            持仓中未考虑缓存。
        """
        # TODO:
        # 如果当前股票列表超过topk（例如权重被风险控制修改），权重将无法正确处理。
        buy_signal_stocks = set(score.sort_values(ascending=False).iloc[: self.topk].index)
        cur_stock_weight = current.get_stock_weight_dict(only_stock=True)

        if len(cur_stock_weight) == 0:
            final_stock_weight = {code: 1 / self.topk for code in buy_signal_stocks}
        else:
            final_stock_weight = copy.deepcopy(cur_stock_weight)
            sold_stock_weight = 0.0
            for stock_id in final_stock_weight:
                if stock_id not in buy_signal_stocks:
                    sw = min(self.max_sold_weight, final_stock_weight[stock_id])
                    sold_stock_weight += sw
                    final_stock_weight[stock_id] -= sw
            if self.buy_method == "first_fill":
                for stock_id in buy_signal_stocks:
                    add_weight = min(
                        max(1 / self.topk - final_stock_weight.get(stock_id, 0), 0.0),
                        sold_stock_weight,
                    )
                    final_stock_weight[stock_id] = final_stock_weight.get(stock_id, 0.0) + add_weight
                    sold_stock_weight -= add_weight
            elif self.buy_method == "average_fill":
                for stock_id in buy_signal_stocks:
                    final_stock_weight[stock_id] = final_stock_weight.get(stock_id, 0.0) + sold_stock_weight / len(
                        buy_signal_stocks
                    )
            else:
                raise ValueError("Buy method not found")
        return final_stock_weight
