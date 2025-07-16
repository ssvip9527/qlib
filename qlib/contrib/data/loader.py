from qlib.data.dataset.loader import QlibDataLoader


class Alpha360DL(QlibDataLoader):
    """用于获取Alpha360的数据集加载器"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        # 注意：
        # Alpha360尝试提供包含原始价格数据的数据集
        # 原始价格数据包括过去60天的价格和成交量
        # 为便于从该数据集学习模型，所有价格和成交量均通过最新价格和成交量数据进行归一化（除以$close、$volume）
        # 因此最新归一化后的$close将为1（名称为CLOSE0），最新归一化后的$volume将为1（名称为VOLUME0）
        # 若执行进一步归一化（如中心化处理），CLOSE0和VOLUME0将变为0。
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class Alpha158DL(QlibDataLoader):
    """获取Alpha158的数据集加载器"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
    ):
        """根据配置创建因子

        config = {
            'kbar': {}, # 是否使用一些硬编码的K线特征
            'price': { # 是否使用原始价格特征
                'windows': [0, 1, 2, 3, 4], # 使用n天前的价格
                'feature': ['OPEN', 'HIGH', 'LOW'] # 要使用的价格字段
            },
            'volume': { # 是否使用原始成交量特征
                'windows': [0, 1, 2, 3, 4], # 使用n天前的成交量
            },
            'rolling': { # 是否使用基于滚动窗口算子的特征
                'windows': [5, 10, 20, 30, 60], # 滚动窗口大小
                'include': ['ROC', 'MA', 'STD'], # 要使用的滚动算子
                # 如果include为None，将使用默认算子
                'exclude': ['RANK'], # 不使用的滚动算子
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # 数据集中的`exclude`是不必要的字段
            # 数据集中的`include`是必要的字段

            def use(x):
                return x not in exclude and (include is None or x in include)

            # 部分因子参考: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # 变动率指标，过去d天的价格变化，除以最新收盘价去除单位影响
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # 简单移动平均线，过去d天的收盘价均值，除以最新收盘价去除单位影响
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # 过去d天收盘价的标准差，除以最新收盘价去除单位影响
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # 过去d天收盘价变化的斜率，除以最新收盘价去除单位影响
                # 例如：过去d天价格每天上涨10美元，则Slope值为10
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # 过去d天线性回归的R平方值，表示趋势的线性程度
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # 过去d天线性回归的残差值，表示过去d天趋势的线性程度
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # 过去d天的最高价，除以最新收盘价去除单位影响
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # 过去d天的最低价，除以最新收盘价去除单位影响
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # 过去d天收盘价的80%分位数，除以最新收盘价去除单位影响
                # 与MIN和MAX指标配合使用
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # 过去d天收盘价的20%分位数，除以最新收盘价去除单位影响
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # 获取当前收盘价在过去d天收盘价中的百分位
                # 表示当前价格相对于过去N天的水平，为移动平均线提供额外信息
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # 表示当前价格在过去d天最高价和最低价之间的位置
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # 当前日期与之前最高价日期之间的天数
                # Aroon指标的一部分 https://www.investopedia.com/terms/a/aroon.asp
                # 该指标衡量一段时间内高点与高点、低点与低点之间的时间间隔
                # 其原理是：强劲的上升趋势会定期出现新高，强劲的下降趋势会定期出现新低
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # 当前日期与之前最低价日期之间的天数
                # Aroon指标的一部分 https://www.investopedia.com/terms/a/aroon.asp
                # 该指标衡量一段时间内高点与高点、低点与低点之间的时间间隔
                # 其原理是：强劲的上升趋势会定期出现新高，强劲的下降趋势会定期出现新低
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # 最高价日期与之后最低价日期之间的时间间隔
                # 较大值表明存在下降动量
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # 收盘价绝对值与对数成交量之间的相关性
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # 价格变化率与成交量变化率之间的相关性
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # 过去d天中价格上涨的天数占比
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # 过去d天中价格下跌的天数占比
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # 过去上涨天数与下跌天数的差值
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # 总收益/绝对总价格变化
                # 类似于RSI指标 https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # 总损失/绝对总价格变化
                # 可从SUMP推导得到：SUMN = 1 - SUMP
                # 类似于RSI指标 https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # 总收益与总损失之间的差异比率
                # 类似于RSI指标 https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # 过去d天成交量的简单移动平均
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # 过去d天成交量的标准差
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # 过去d天成交量加权的价格变动波动率
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # 总成交量增加/绝对总成交量变化
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # 总成交量增加/绝对总成交量变化
                # 可从VSUMP推导得到：VSUMN = 1 - VSUMP
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # 总成交量增加与减少之间的差异比率
                # 成交量版的RSI指标
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names
