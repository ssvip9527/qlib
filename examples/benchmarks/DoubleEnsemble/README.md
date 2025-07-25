# DoubleEnsemble集成框架
* DoubleEnsemble是一种集成框架，它利用基于学习轨迹的样本重加权和基于洗牌的特征选择，来解决信噪比低和特征数量增加的问题。该框架基于每个样本的训练动态识别关键样本，并通过洗牌方式基于每个特征的消融影响提取关键特征。该模型适用于多种基础模型，能够提取复杂模式，同时减轻金融市场预测中的过拟合和不稳定性问题。
* 本代码在Qlib中的实现由我们自行完成。
* 论文：DoubleEnsemble：一种基于样本重加权和特征选择的金融数据分析新集成方法 [https://arxiv.org/pdf/2010.01265.pdf](https://arxiv.org/pdf/2010.01265.pdf)。