# 时间相关任务调度序列学习
### 背景
近年来，序列学习引起了机器学习社区的广泛研究关注。在许多应用中，一个序列学习任务通常与多个时间相关的辅助任务相关联，这些辅助任务在使用多少输入信息或预测哪个未来步骤方面有所不同。在股票趋势预测中，如图1所示，可以预测股票在不同未来天数（例如明天、后天）的价格。本文提出了一个利用这些时间相关任务相互帮助的框架。

### 方法
鉴于通常存在多个时间相关任务，关键挑战在于在训练过程中选择哪些任务以及何时使用这些任务。本工作为序列学习引入了一个可学习的任务调度器，该调度器在训练过程中自适应地选择时间相关任务。调度器访问模型状态和当前训练数据（例如当前小批量数据），并选择最佳辅助任务来帮助主任务的训练。调度器和主任务模型通过双层优化联合训练：调度器的训练目标是最大化模型的验证性能，而模型的训练目标是在调度器的指导下最小化训练损失。该过程如图2所示。

<p align="center"> 
<img src="workflow.png"/>
</p>

在步骤<img src="https://latex.codecogs.com/png.latex?s" title="s" />，给定训练数据<img src="https://latex.codecogs.com/png.latex?x_s,y_s" title="x_s,y_s" />，调度器<img src="https://latex.codecogs.com/png.latex?\varphi" title="\varphi" />选择合适的任务<img src="https://latex.codecogs.com/png.latex?T_{i_s}" title="T_{i_s}" />（绿色实线）来更新模型<img src="https://latex.codecogs.com/png.latex?f" title="f" />（蓝色实线）。经过<img src="https://latex.codecogs.com/png.latex?S" title="S" />步后，我们在验证集上评估模型<img src="https://latex.codecogs.com/png.latex?f" title="f" />并更新调度器<img src="https://latex.codecogs.com/png.latex?\varphi" title="\varphi" />（绿色虚线）。

### 实验
由于数据版本和Qlib版本不同，论文中实验设置的原始数据和数据预处理方法与现有Qlib版本中的实验设置有所不同。因此，我们根据两种设置提供了两个版本的代码：1）可用于复现实验结果的[代码](https://github.com/lwwang1995/tcts)；2）当前Qlib基线中的[代码](https://github.com/ssvip9527/qlib/blob/main/qlib/contrib/model/pytorch_tcts.py)。

#### 设置1
* 数据集：我们使用[沪深300](http://www.csindex.com.cn/en/indices/index-detail/000300) 300只股票2008年1月1日至2020年8月1日的历史交易数据。根据交易时间将数据分为训练集（2008/01/01-2013/12/31）、验证集（2014/01/01-2015/12/31）和测试集（2016/01/01-2020/08/01）。

* 主任务<img src="https://latex.codecogs.com/png.latex?T_k" title="T_k" />指预测股票<img src="https://latex.codecogs.com/png.latex?i" title="i" />的收益，定义如下：
<div align=center>
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r_{i}^{t,k}&space;=&space;\frac{price_i^{t&plus;k}}{price_i^{t&plus;k-1}}-1" title="r_{i}^{t,k} = \frac{price_i^{t+k}}{price_i^{t+k-1}}-1" />
</div>

* 时间相关任务集<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_k&space;=&space;\{T_1,&space;T_2,&space;...&space;,&space;T_k\}" title="\mathcal{T}_k = \{T_1, T_2, ... , T_k\}" />，本文中，<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_3" title="\mathcal{T}_3" />、<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_5" title="\mathcal{T}_5" />和<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_{10}" title="\mathcal{T}_{10}" />分别用于<img src="https://latex.codecogs.com/png.latex?T_1" title="T_1" />、<img src="https://latex.codecogs.com/png.latex?T_2" title="T_2" />和<img src="https://latex.codecogs.com/png.latex?T_3" title="T_3" />。

#### 设置2
* 数据集：我们使用[沪深300](http://www.csindex.com.cn/en/indices/index-detail/000300) 300只股票2008年1月1日至2020年8月1日的历史交易数据。根据交易时间将数据分为训练集（2008/01/01-2014/12/31）、验证集（2015/01/01-2016/12/31）和测试集（2017/01/01-2020/08/01）。

* 主任务<img src="https://latex.codecogs.com/png.latex?T_k" title="T_k" />指预测股票<img src="https://latex.codecogs.com/png.latex?i" title="i" />的收益，定义如下：
<div align=center>
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r_{i}^{t,k}&space;=&space;\frac{price_i^{t&plus;1&plus;k}}{price_i^{t&plus;1}}-1" title="r_{i}^{t,k} = \frac{price_i^{t+1+k}}{price_i^{t+1}}-1" />
</div>

* 在Qlib基线中，<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_3" title="\mathcal{T}_3" />用于<img src="https://latex.codecogs.com/png.latex?T_1" title="T_1" />。

### 实验结果
设置1的实验结果可在[论文](http://proceedings.mlr.press/v139/wu21e/wu21e.pdf)中找到，设置2的实验结果可在本[页面](https://github.com/ssvip9527/qlib/tree/main/examples/benchmarks)中找到。