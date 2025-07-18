[![Python Versions](https://img.shields.io/pypi/pyversions/pyqlib.svg?logo=python&logoColor=white)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![PypI Versions](https://img.shields.io/pypi/v/pyqlib)](https://pypi.org/project/pyqlib/#history)
[![Upload Python Package](https://github.com/ssvip9527/qlib/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/pyqlib/)
[![Github Actions Test Status](https://github.com/ssvip9527/qlib/workflows/Test/badge.svg?branch=main)](https://github.com/ssvip9527/qlib/actions)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)
[![Join the chat at https://gitter.im/Microsoft/qlib](https://badges.gitter.im/Microsoft/qlib.svg)](https://gitter.im/Microsoft/qlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## :newspaper: **最新消息！** &nbsp;   :sparkling_heart: 

最近发布的特性

### 介绍 <a href="https://github.com/microsoft/RD-Agent"><img src="docs/_static/img/rdagent_logo.png" alt="RD_Agent" style="height: 2em"></a>: 基于LLM的自主演化代理，用于工业数据驱动的研发

我们很高兴宣布发布 **RD-Agent**📢，这是一个强大的工具，支持量化投资研发中的自动化因子挖掘和模型优化。

RD-Agent 现在已在 [GitHub](https://github.com/microsoft/RD-Agent) 上可用，我们欢迎您的星标🌟！

要了解更多信息，请访问我们的 [♾️演示页面](https://rdagent.azurewebsites.net/)。在这里，您将找到英文和中文演示视频，帮助您更好地理解 RD-Agent 的场景和用法。

我们为您准备了几个演示视频：
| 场景 | 演示视频 (English) | 演示视频 (中文) |
| --                      | ------    | ------    |
| 量化因子挖掘 | [链接](https://rdagent.azurewebsites.net/factor_loop?lang=en) | [链接](https://rdagent.azurewebsites.net/factor_loop?lang=zh) |
| 从报告中量化因子挖掘 | [链接](https://rdagent.azurewebsites.net/report_factor?lang=en) | [链接](https://rdagent.azurewebsites.net/report_factor?lang=zh) |
| 量化模型优化 | [链接](https://rdagent.azurewebsites.net/model_loop?lang=en) | [链接](https://rdagent.azurewebsites.net/model_loop?lang=zh) |

- 📃**论文**: [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
- 👾**代码**: https://github.com/microsoft/RD-Agent/
```BibTeX
@misc{li2025rdagentquant,
    title={R\&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization},
    author={Yuante Li and Xu Yang and Xiao Yang and Minrui Xu and Xisen Wang and Weiqing Liu and Jiang Bian},
    year={2025},
    eprint={2505.15155},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
![image](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

***

| 特性 | 状态 |
| --                      | ------    |
| [R&D-Agent-Quant](https://arxiv.org/abs/2505.15155) 已发布 | 将 R&D-Agent 应用于 Qlib 用于量化交易 | 
| BPQP 用于端到端学习 | 📈即将到来！([审核中](https://github.com/ssvip9527/qlib/pull/1863)) |
| 🔥LLM驱动的自动量化工厂🔥 | 🚀 于2024年8月8日在 [♾️RD-Agent](https://github.com/microsoft/RD-Agent) 中发布 |
| KRNN 和 Sandwich 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/1414/) 于2023年5月26日 |
| 发布 Qlib v0.9.0 | :octocat: [已发布](https://github.com/ssvip9527/qlib/releases/tag/v0.9.0) 于2022年12月9日 |
| RL 学习框架 | :hammer: :chart_with_upwards_trend: 于2022年11月10日发布。 [#1332](https://github.com/ssvip9527/qlib/pull/1332), [#1322](https://github.com/ssvip9527/qlib/pull/1322), [#1316](https://github.com/ssvip9527/qlib/pull/1316),[#1299](https://github.com/ssvip9527/qlib/pull/1299),[#1263](https://github.com/ssvip9527/qlib/pull/1263), [#1244](https://github.com/ssvip9527/qlib/pull/1244), [#1169](https://github.com/ssvip9527/qlib/pull/1169), [#1125](https://github.com/ssvip9527/qlib/pull/1125), [#1076](https://github.com/ssvip9527/qlib/pull/1076)|
| HIST 和 IGMTF 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/1040) 于2022年4月10日 |
| Qlib [notebook 教程](https://github.com/ssvip9527/qlib/tree/main/examples/tutorial) | 📖 [已发布](https://github.com/ssvip9527/qlib/pull/1037) 于2022年4月7日 | 
| Ibovespa 指数数据 | :rice: [已发布](https://github.com/ssvip9527/qlib/pull/990) 于2022年4月6日 |
| Point-in-Time 数据库 | :hammer: [已发布](https://github.com/ssvip9527/qlib/pull/343) 于2022年3月10日 |
| Arctic Provider Backend & Orderbook 数据示例 | :hammer: [已发布](https://github.com/ssvip9527/qlib/pull/744) 于2022年1月17日 |
| 基于元学习的框架 & DDG-DA  | :chart_with_upwards_trend:  :hammer: [已发布](https://github.com/ssvip9527/qlib/pull/743) 于2022年1月10日 | 
| 基于规划的投资组合优化 | :hammer: [已发布](https://github.com/ssvip9527/qlib/pull/754) 于2021年12月28日 | 
| 发布 Qlib v0.8.0 | :octocat: [已发布](https://github.com/ssvip9527/qlib/releases/tag/v0.8.0) 于2021年12月8日 |
| ADD 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/704) 于2021年11月22日 |
| ADARNN 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/689) 于2021年11月14日 |
| TCN 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/668) 于2021年11月4日 |
| 嵌套决策框架 | :hammer: [已发布](https://github.com/ssvip9527/qlib/pull/438) 于2021年10月1日。 [示例](https://github.com/ssvip9527/qlib/blob/main/examples/nested_decision_execution/workflow.py) 和 [文档](https://qlib.readthedocs.io/en/latest/component/highfreq.html) |
| Temporal Routing Adaptor (TRA) | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/531) 于2021年7月30日 |
| Transformer & Localformer | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/508) 于2021年7月22日 |
| 发布 Qlib v0.7.0 | :octocat: [已发布](https://github.com/ssvip9527/qlib/releases/tag/v0.7.0) 于2021年7月12日 |
| TCTS 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/491) 于2021年7月1日 |
| 在线服务和自动模型滚动 | :hammer:  [已发布](https://github.com/ssvip9527/qlib/pull/290) 于2021年5月17日 | 
| DoubleEnsemble 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/286) 于2021年3月2日 | 
| 高频数据处理示例 | :hammer: [已发布](https://github.com/ssvip9527/qlib/pull/257) 于2021年2月5日  |
| 高频交易示例 | :chart_with_upwards_trend: [部分代码已发布](https://github.com/ssvip9527/qlib/pull/227) 于2021年1月28日  | 
| 高频数据(1min) | :rice: [已发布](https://github.com/ssvip9527/qlib/pull/221) 于2021年1月27日 |
| Tabnet 模型 | :chart_with_upwards_trend: [已发布](https://github.com/ssvip9527/qlib/pull/205) 于2021年1月22日 |

2021年之前发布的特性未在此列出。

<p align="center">
  <img src="docs/_static/img/logo/1.png" />
</p>

Qlib 是一个开源的、面向AI的量化投资平台，旨在利用AI技术在量化投资中实现潜力、赋能研究并创造价值，从探索想法到实施生产。Qlib 支持多种机器学习建模范式，包括监督学习、市场动态建模和强化学习。

越来越多的 SOTA 量化研究工作/论文在 Qlib 中发布，以协作解决量化投资中的关键挑战。例如，1) 使用监督学习从丰富且异构的金融数据中挖掘市场的复杂非线性模式，2) 使用自适应概念漂移技术建模金融市场的动态性质，3) 使用强化学习建模连续投资决策并帮助投资者优化交易策略。

它包含数据处理、模型训练、回测的完整ML管道；并覆盖量化投资的整个链条：alpha 寻求、风险建模、投资组合优化和订单执行。
更多细节，请参考我们的论文 ["Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189)。


<table>
  <tbody>
    <tr>
      <th>框架、教程、数据 & DevOps</th>
      <th>量化研究中的主要挑战 & 解决方案</th>
    </tr>
    <tr>
      <td>
        <li><a href="#plans"><strong>计划</strong></a></li>
        <li><a href="#framework-of-qlib">Qlib 框架</a></li>
        <li><a href="#quick-start">快速开始</a></li>
          <ul dir="auto">
            <li type="circle"><a href="#installation">安装</a> </li>
            <li type="circle"><a href="#data-preparation">数据准备</a></li>
            <li type="circle"><a href="#auto-quant-research-workflow">自动量化研究工作流</a></li>
            <li type="circle"><a href="#building-customized-quant-research-workflow-by-code">通过代码构建自定义量化研究工作流</a></li></ul>
        <li><a href="#quant-dataset-zoo"><strong>量化数据集动物园</strong></a></li>
        <li><a href="#learning-framework">学习框架</a></li>
        <li><a href="#more-about-qlib">关于 Qlib 的更多信息</a></li>
        <li><a href="#offline-mode-and-online-mode">离线模式和在线模式</a>
        <ul>
          <li type="circle"><a href="#performance-of-qlib-data-server">Qlib 数据服务器的性能</a></li></ul>
        <li><a href="#related-reports">相关报告</a></li>
        <li><a href="#contact-us">联系我们</a></li>
        <li><a href="#contributing">贡献</a></li>
      </td>
      <td valign="baseline">
        <li><a href="#main-challenges--solutions-in-quant-research">量化研究中的主要挑战 & 解决方案</a>
          <ul>
            <li type="circle"><a href="#forecasting-finding-valuable-signalspatterns">预测：找到有价值的信号/模式</a>
              <ul>
                <li type="disc"><a href="#quant-model-paper-zoo"><strong>量化模型 (论文) 动物园</strong></a>
                  <ul>
                    <li type="circle"><a href="#run-a-single-model">运行单个模型</a></li>
                    <li type="circle"><a href="#run-multiple-models">运行多个模型</a></li>
                  </ul>
                </li>
              </ul>
            </li>
          <li type="circle"><a href="#adapting-to-market-dynamics">适应市场动态</a></li>
          <li type="circle"><a href="#reinforcement-learning-modeling-continuous-decisions">强化学习：建模连续决策</a></li>
          </ul>
        </li>
      </td>
    </tr>
  </tbody>
</table>

# 计划
正在开发的新特性（按预计发布时间排序）。
您的反馈对这些特性非常重要。
<!-- | Feature                        | Status      | -->
<!-- | --                      | ------    | -->

# Qlib 框架

<div style="align: center">
<img src="docs/_static/img/framework-abstract.jpg" />
</div>

Qlib 的高层框架如上所示（用户可以在深入细节时找到 Qlib 设计的[详细框架](https://qlib.readthedocs.io/en/latest/introduction/introduction.html#framework)）。
组件被设计为松散耦合的模块，每个组件都可以独立使用。

Qlib 提供强大的基础设施来支持量化研究。[数据](https://qlib.readthedocs.io/en/latest/component/data.html) 始终是重要的一部分。
设计了一个强大的学习框架来支持不同的学习范式（例如 [强化学习](https://qlib.readthedocs.io/en/latest/component/rl.html), [监督学习](https://qlib.readthedocs.io/en/latest/component/workflow.html#model-section)）和不同级别的模式（例如 [市场动态建模](https://qlib.readthedocs.io/en/latest/component/meta.html)）。
通过建模市场，[交易策略](https://qlib.readthedocs.io/en/latest/component/strategy.html) 将生成将被执行的交易决策。不同级别或粒度的多个交易策略和执行器可以[嵌套以一起优化和运行](https://qlib.readthedocs.io/en/latest/component/highfreq.html)。
最后，将提供全面的[分析](https://qlib.readthedocs.io/en/latest/component/report.html)，并且模型可以以低成本[在线服务](https://qlib.readthedocs.io/en/latest/component/online.html)。


# 快速开始

这个快速开始指南试图演示
1. 使用 _Qlib_ 构建完整的量化研究工作流并尝试您的想法非常容易。
2. 尽管使用*公共数据*和*简单模型*，机器学习技术在实际量化投资中**工作得非常好**。

这里是一个快速 **[演示](https://terminalizer.com/view/3f24561a4470)** 显示如何安装 ``Qlib``，并使用 ``qrun`` 运行 LightGBM。**但是**，请确保您已经按照[说明](#data-preparation)准备好数据。


## 安装

此表展示了 `Qlib` 支持的 Python 版本：
|               | 使用 pip 安装      | 从源代码安装  |        绘图        |
| ------------- |:---------------------:|:--------------------:|:------------------:|
| Python 3.8    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.9    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.10   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.11   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.12   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |

**注意**： 
1. 建议使用 **Conda** 管理您的 Python 环境。在某些情况下，在 `conda` 环境之外使用 Python 可能会导致缺少头文件，从而导致某些包的安装失败。
2. 请注意，在 Python 3.6 中安装 cython 会导致从源代码安装 ``Qlib`` 时出现一些错误。如果用户在他们的机器上使用 Python 3.6，建议*升级* Python 到 3.8 或更高版本，或者使用 `conda` 的 Python 从源代码安装 ``Qlib``。

### 使用 pip 安装
用户可以按照以下命令轻松使用 pip 安装 ``Qlib``。

```bash
  pip install pyqlib
```

**注意**：pip 将安装最新的稳定版 qlib。但是，qlib 的 main 分支处于活跃开发中。如果您想测试 main 分支中的最新脚本或函数。请使用以下方法安装 qlib。

### 从源代码安装
用户也可以按照以下步骤通过源代码安装最新的开发版 ``Qlib``：

* 在从源代码安装 ``Qlib`` 之前，用户需要安装一些依赖项：

  ```bash
  pip install numpy
  pip install --upgrade cython
  ```

* 克隆仓库并安装 ``Qlib`` 如下。
    ```bash
    git clone https://github.com/ssvip9527/qlib.git && cd qlib
    pip install .  # `pip install -e .[dev]` 推荐用于开发。检查 docs/developer/code_standard_and_dev_guide.rst 中的细节
    ```

**提示**：如果您在环境中安装 `Qlib` 或运行示例失败，比较您的步骤和 [CI 工作流](.github/workflows/test_qlib_from_source.yml) 可能有助于您找到问题。

**Mac 提示**：如果您使用的是带 M1 的 Mac，您可能会在为 LightGBM 构建 wheel 时遇到问题，这是由于缺少 OpenMP 的依赖项。为了解决问题，首先使用 ``brew install libomp`` 安装 openmp，然后运行 ``pip install .`` 以成功构建它。

## 数据准备
❗ 由于更严格的数据安全策略。官方数据集暂时禁用。您可以尝试社区贡献的[此数据源](https://github.com/chenditc/investment_data/releases)。
这里是一个下载最新数据的示例。
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

下面的官方数据集将在不久的将来恢复。


----

通过运行以下代码加载和准备数据：

### 使用模块获取
  ```bash
  # 获取 1d 数据
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取 1min 数据
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

### 从源获取

  ```bash
  # 获取 1d 数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取 1min 数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

此数据集是由 [crawler scripts](scripts/data_collector/) 收集的公共数据创建的，这些脚本已在同一仓库中发布。
用户可以使用它创建相同的数据集。[数据集描述](https://github.com/ssvip9527/qlib/tree/main/scripts/data_collector#description-of-dataset)

*请**注意**数据是从 [Yahoo Finance](https://finance.yahoo.com/lookup) 收集的，数据可能不完美。
我们建议用户如果有高质量数据集，请准备自己的数据。更多信息，用户可以参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*。

### 自动更新每日频率数据 (从 yahoo finance)
  > 如果用户只想在历史数据上尝试他们的模型和策略，此步骤是*可选*的。
  > 
  > 建议用户手动更新一次数据 (--trading_date 2021-05-25) 然后设置为自动更新。
  >
  > **注意**：用户无法基于 Qlib 提供的离线数据（一些字段被移除以减少数据大小）增量更新数据。用户应使用 [yahoo collector](https://github.com/ssvip9527/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance) 从头下载 Yahoo 数据，然后增量更新它。
  > 
  > 更多信息，请参考：[yahoo collector](https://github.com/ssvip9527/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)

  * 每个交易日自动更新数据到 "qlib" 目录 (Linux)
      * 使用 *crontab*: `crontab -e`
      * 设置定时任务：

        ```
        * * * * 1-5 python <script path> update_data_to_bin --qlib_data_1d_dir <user data dir>
        ```
        * **script path**: *scripts/data_collector/yahoo/collector.py*

  * 手动更新数据
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
      ```
      * *trading_date*: 交易日开始
      * *end_date*: 交易日结束（不包括）

### 检查数据健康状况
  * 我们提供了一个脚本来检查数据的健康状况，您可以运行以下命令来检查数据是否健康。
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data
    ```
  * 当然，您也可以添加一些参数来调整测试结果，例如这样。
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data --missing_data_num 30055 --large_step_threshold_volume 94485 --large_step_threshold_price 20
    ```
  * 如果您想了解更多关于 `check_data_health` 的信息，请参考[文档](https://qlib.readthedocs.io/en/latest/component/data.html#checking-the-health-of-the-data)。

<!-- 
- Run the initialization code and get stock data:

  ```python
  import qlib
  from qlib.data import D
  from qlib.constant import REG_CN

  # Initialization
  mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
  qlib.init(mount_path=mount_path, region=REG_CN)

  # Get stock data by Qlib
  # Load trading calendar with the given time range and frequency
  print(D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2])

  # Parse a given market name into a stockpool config
  instruments = D.instruments('csi500')
  print(D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6])

  # Load features of certain instruments in given time range
  instruments = ['SH600000']
  fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
  print(D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head())
  ```
 -->

## Docker 镜像
1. 从 Docker Hub 仓库拉取镜像
```bash
docker pull pyqlib/qlib_image_stable:stable
```
2. 启动新的 Docker 容器
```bash
docker run -it --name <容器名> -v <本地挂载目录>:/app qlib_image_stable
```
3. 此时你已进入 docker 环境，可以运行 qlib 脚本。例如：
    ```bash
>>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
>>> python qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```
4. 退出容器
```bash
>>> exit
```
5. 重启容器
```bash
docker start -i -a <容器名>
```
6. 停止容器
    ```bash
docker stop <容器名>
```
7. 删除容器
```bash
docker rm <容器名>
```
8. 更多信息请参考[文档](https://qlib.readthedocs.io/en/latest/developer/how_to_build_image.html)。

## 自动化量化研究工作流
Qlib 提供了名为 `qrun` 的工具，可自动运行完整的量化研究工作流（包括数据集构建、模型训练、回测和评估）。你可以按照以下步骤启动自动化量化研究工作流，并进行图形化报告分析：

1. 量化研究工作流：使用 lightgbm 的工作流配置文件运行 `qrun`（如 [workflow_config_lightgbm_Alpha158.yaml](examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)）。
    ```bash
      cd examples  # 避免在包含 `qlib` 的目录下运行程序
      qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    如果用户希望在调试模式下使用 `qrun`，请使用以下命令：
    ```bash
    python -m pdb qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    `qrun` 的结果如下，更多解释请参考[文档](https://qlib.readthedocs.io/en/latest/component/strategy.html#result)。

    ```bash

    '以下为超额收益（不含成本）的分析结果。'
                           risk
    mean               0.000708
    std                0.005626
    annualized_return  0.178316
    information_ratio  1.996555
    max_drawdown      -0.081806
    '以下为超额收益（含成本）的分析结果。'
                           risk
    mean               0.000512
    std                0.005626
    annualized_return  0.128982
    information_ratio  1.444287
    max_drawdown      -0.091078
    ```
    有关 `qrun` 和 [工作流](https://qlib.readthedocs.io/en/latest/component/workflow.html) 的详细文档请参考链接。

2. 图形化报告分析：首先运行 `python -m pip install .[analysis]` 安装所需依赖，然后用 `jupyter notebook` 打开并运行 `examples/workflow_by_code.ipynb`，即可获得图形化报告。
    - 预测信号（模型预测）分析
      - 分组累计收益
      ![Cumulative Return](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/analysis_model_cumulative_return.png)
      - 收益分布
      ![long_short](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/analysis_model_long_short.png)
      - 信息系数（IC）
      ![Information Coefficient](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/analysis_model_IC.png)
      ![Monthly IC](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/analysis_model_monthly_IC.png)
      ![IC](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/analysis_model_NDQ.png)
      - 预测信号（模型预测）的自相关性
      ![Auto Correlation](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/analysis_model_auto_correlation.png)

    - 投资组合分析
      - 回测收益
      ![Report](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/analysis/report.png)
      <!-- 
      - Score IC
      ![Score IC](docs/_static/img/score_ic.png)
      - Cumulative Return
      ![Cumulative Return](docs/_static/img/cumulative_return.png)
      - Risk Analysis
      ![Risk Analysis](docs/_static/img/risk_analysis.png)
      - Rank Label
      ![Rank Label](docs/_static/img/rank_label.png)
      -->
   - 上述结果的[详细解释](https://qlib.readthedocs.io/en/latest/component/report.html)

## 通过代码构建自定义量化研究工作流
自动化工作流可能并不适用于所有量化研究者的研究流程。为支持灵活的量化研究工作流，Qlib 还提供了模块化接口，允许研究者通过代码自定义工作流。[这里](examples/workflow_by_code.ipynb)有一个通过代码自定义量化研究工作流的演示。

# 量化研究中的主要挑战与解决方案
量化投资是一个非常独特的场景，存在许多关键挑战需要解决。
目前，Qlib 针对其中一些挑战提供了解决方案。

## 预测：寻找有价值的信号/模式
准确预测股票价格趋势对于构建盈利投资组合至关重要。
然而，金融市场中存在大量不同格式的数据，这使得构建预测模型变得具有挑战性。

越来越多关注在复杂金融数据中挖掘有价值信号/模式的 SOTA 量化研究工作/论文在 `Qlib` 中发布。


### [量化模型（论文）动物园](examples/benchmarks)

以下是基于 `Qlib` 构建的模型列表：
- [基于 XGBoost 的 GBDT (Tianqi Chen, et al. KDD 2016)](examples/benchmarks/XGBoost/)
- [基于 LightGBM 的 GBDT (Guolin Ke, et al. NIPS 2017)](examples/benchmarks/LightGBM/)
- [基于 Catboost 的 GBDT (Liudmila Prokhorenkova, et al. NIPS 2018)](examples/benchmarks/CatBoost/)
- [基于 pytorch 的 MLP](examples/benchmarks/MLP/)
- [基于 pytorch 的 LSTM (Sepp Hochreiter, et al. Neural computation 1997)](examples/benchmarks/LSTM/)
- [基于 pytorch 的 GRU (Kyunghyun Cho, et al. 2014)](examples/benchmarks/GRU/)
- [基于 pytorch 的 ALSTM (Yao Qin, et al. IJCAI 2017)](examples/benchmarks/ALSTM)
- [基于 pytorch 的 GATs (Petar Velickovic, et al. 2017)](examples/benchmarks/GATs/)
- [基于 pytorch 的 SFM (Liheng Zhang, et al. KDD 2017)](examples/benchmarks/SFM/)
- [基于 tensorflow 的 TFT (Bryan Lim, et al. International Journal of Forecasting 2019)](examples/benchmarks/TFT/)
- [基于 pytorch 的 TabNet (Sercan O. Arik, et al. AAAI 2019)](examples/benchmarks/TabNet/)
- [基于 LightGBM 的 DoubleEnsemble (Chuheng Zhang, et al. ICDM 2020)](examples/benchmarks/DoubleEnsemble/)
- [基于 pytorch 的 TCTS (Xueqing Wu, et al. ICML 2021)](examples/benchmarks/TCTS/)
- [基于 pytorch 的 Transformer (Ashish Vaswani, et al. NeurIPS 2017)](examples/benchmarks/Transformer/)
- [基于 pytorch 的 Localformer (Juyong Jiang, et al.)](examples/benchmarks/Localformer/)
- [基于 pytorch 的 TRA (Hengxu, Dong, et al. KDD 2021)](examples/benchmarks/TRA/)
- [基于 pytorch 的 TCN (Shaojie Bai, et al. 2018)](examples/benchmarks/TCN/)
- [基于 pytorch 的 ADARNN (YunTao Du, et al. 2021)](examples/benchmarks/ADARNN/)
- [基于 pytorch 的 ADD (Hongshun Tang, et al.2020)](examples/benchmarks/ADD/)
- [基于 pytorch 的 IGMTF (Wentao Xu, et al.2021)](examples/benchmarks/IGMTF/)
- [基于 pytorch 的 HIST (Wentao Xu, et al.2021)](examples/benchmarks/HIST/)
- [基于 pytorch 的 KRNN](examples/benchmarks/KRNN/)
- [基于 pytorch 的 Sandwich](examples/benchmarks/Sandwich/)

欢迎提交新的量化模型 PR。

每个模型在 `Alpha158` 和 `Alpha360` 数据集上的表现可在[这里](examples/benchmarks/README.md)查看。

### 运行单个模型
上述所有模型均可通过 ``Qlib`` 运行。用户可在 [benchmarks](examples/benchmarks) 文件夹中找到我们提供的配置文件及模型详情，更多信息可在上述模型文件中获取。

`Qlib` 提供三种不同方式运行单个模型，用户可根据实际情况选择：
- 使用上文提到的 `qrun` 工具，通过配置文件运行模型工作流。
- 基于 `examples` 文件夹下的 [workflow_by_code.py](examples/workflow_by_code.py) 创建 `workflow_by_code` Python 脚本。

- 使用 `examples` 文件夹下的 [`run_all_model.py`](examples/run_all_model.py) 脚本运行模型。例如：`python run_all_model.py run --models=lightgbm`，其中 `--models` 参数可指定上述任意模型（可用模型见 [benchmarks](examples/benchmarks/)）。更多用法请参考文件的 [docstring](examples/run_all_model.py)。
    - **注意**：每个基线模型依赖的环境不同，请确保你的 Python 版本与要求一致（如 TFT 仅支持 Python 3.6~3.7，因 `tensorflow==1.15.0` 限制）。

### 运行多个模型
`Qlib` 还提供了 [`run_all_model.py`](examples/run_all_model.py) 脚本，可多次迭代运行多个模型。（**注意**：该脚本目前仅支持 *Linux*，其他操作系统未来会支持。此外，目前不支持同一模型多次并行运行，后续开发将修复此问题。）

该脚本会为每个模型创建独立虚拟环境，训练完成后自动删除，仅保留实验结果如 `IC` 和 `backtest`。

例如，运行所有模型 10 次：
```python
python run_all_model.py run 10
```

也可通过 API 一次性运行指定模型。更多用法请参考文件的 [docstring](examples/run_all_model.py)。

### 兼容性变更
在 `pandas` 中，`group_key` 是 `groupby` 方法的参数之一。从 1.5 版本到 2.0 版本，`group_key` 的默认值由 `无默认` 变为 `True`，这会导致 qlib 运行时报错。因此我们设置了 `group_key=False`，但不能保证所有程序都能正确运行，包括：
* qlib\examples\rl_order_execution\scripts\gen_training_orders.py
* qlib\examples\benchmarks\TRA\src\dataset.MTSDatasetH.py
* qlib\examples\benchmarks\TFT\tft.py



## [适应市场动态](examples/benchmarks_dynamic)

由于金融市场环境的非平稳性，数据分布在不同阶段可能发生变化，这导致基于训练数据构建的模型在未来测试数据上的表现下降。因此，使预测模型/策略适应市场动态对于其性能至关重要。

以下是基于 `Qlib` 构建的解决方案列表：
- [滚动再训练](examples/benchmarks_dynamic/baseline/)
- [基于 pytorch 的 DDG-DA (Wendi, et al. AAAI 2022)](examples/benchmarks_dynamic/DDG-DA/)

## 强化学习：建模连续决策
Qlib 现已支持强化学习功能，用于建模连续投资决策。该功能通过与环境的交互学习，帮助投资者优化交易策略，以最大化累计收益。

以下是基于不同场景的 Qlib 强化学习解决方案列表：

### [订单执行中的强化学习](examples/rl_order_execution)
[这里](https://qlib.readthedocs.io/en/latest/component/rl/overall.html#order-execution)有该场景的介绍，所有方法的对比见[此处](examples/rl_order_execution)。
- [TWAP](examples/rl_order_execution/exp_configs/backtest_twap.yml)
- [PPO: "An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization", IJCAL 2020](examples/rl_order_execution/exp_configs/backtest_ppo.yml)
- [OPDS: "Universal Trading for Order Execution with Oracle Policy Distillation", AAAI 2021](examples/rl_order_execution/exp_configs/backtest_opds.yml)

# 量化数据集动物园
数据集在量化研究中扮演着非常重要的角色。以下是基于 `Qlib` 构建的数据集列表：

| 数据集                                    | 美国市场 | 中国市场 |
| --                                         | --        | --           |
| [Alpha360](./qlib/contrib/data/handler.py) |  √        |  √           |
| [Alpha158](./qlib/contrib/data/handler.py) |  √        |  √           |

[这里](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)有使用 `Qlib` 构建数据集的教程。
欢迎提交新的量化数据集 PR。


# 学习框架
Qlib 高度可定制，许多组件都可学习。
可学习组件包括 `预测模型` 和 `交易代理` 的实例。它们基于 `学习框架` 层进行训练，然后应用于 `工作流` 层的多个场景。
学习框架也会利用 `工作流` 层（如共享 `信息提取器`，基于 `执行环境` 创建环境等）。

根据学习范式，可分为强化学习和监督学习：
- 监督学习的详细文档见[这里](https://qlib.readthedocs.io/en/latest/component/model.html)。
- 强化学习的详细文档见[这里](https://qlib.readthedocs.io/en/latest/component/rl.html)。Qlib 的 RL 学习框架利用 `工作流` 层的 `执行环境` 创建环境，支持 `NestedExecutor`，可实现多层次策略/模型/代理的联合优化（如针对特定投资组合管理策略优化订单执行策略）。


# 关于 Qlib 的更多信息
如果你想快速了解 qlib 最常用的组件，可以尝试 [notebooks 示例](examples/tutorial/)。

详细文档见 [docs](docs/)。
构建 HTML 格式文档需安装 [Sphinx](http://www.sphinx-doc.org) 及 readthedocs 主题。
```bash
cd docs/
conda install sphinx sphinx_rtd_theme -y
# 或者用 pip 安装
# pip install sphinx sphinx_rtd_theme
make html
```
你也可以直接在线查看[最新文档](http://qlib.readthedocs.io/)。

Qlib 正在持续开发中，开发计划见 [github 项目](https://github.com/ssvip9527/qlib/projects/1)。



# 离线与在线模式
Qlib 的数据服务器可部署为“离线”模式或“在线”模式，默认模式为离线模式。

在“离线”模式下，数据会本地部署。

在“在线”模式下，数据作为共享数据服务部署，数据及其缓存会被所有客户端共享。由于缓存命中率更高，数据检索性能有望提升，同时也会节省磁盘空间。在线模式文档见 [Qlib-Server](https://qlib-server.readthedocs.io/)，可通过 [Azure CLI 脚本一键部署](https://qlib-server.readthedocs.io/en/latest/build.html#one-click-deployment-in-azure)。在线数据服务器源码见 [Qlib-Server 仓库](https://github.com/ssvip9527/qlib-server)。

## Qlib 数据服务器性能
数据处理性能对 AI 等数据驱动方法至关重要。作为面向 AI 的平台，Qlib 提供了数据存储与处理的解决方案。为展示 Qlib 数据服务器的性能，我们与其他数据存储方案进行了对比。

我们通过同一任务评估多种存储方案性能：从股票市场（2007-2020 年，800 只股票每日 OHLCV 数据）生成包含 14 个特征/因子的因子数据集，涉及数据查询与处理。

|                         | HDF5      | MySQL     | MongoDB   | InfluxDB  | Qlib -E -D  | Qlib +E -D   | Qlib +E +D  |
| --                      | ------    | ------    | --------  | --------- | ----------- | ------------ | ----------- |
| 单核总耗时（秒）        | 184.4±3.7 | 365.3±7.5 | 253.6±6.7 | 368.2±3.6 | 147.0±8.8   | 47.6±1.0     | **7.4±0.3** |
| 64 核总耗时（秒）       |           |           |           |           | 8.8±0.6     | **4.2±0.2**  |             |
* `+(-)E` 表示有（无）`ExpressionCache`
* `+(-)D` 表示有（无）`DatasetCache`

大多数通用数据库加载数据耗时较长。深入底层实现后发现，通用数据库方案中数据需经过多层接口和不必要的格式转换，极大拖慢了数据加载速度。
Qlib 数据以紧凑格式存储，便于高效组合为科学计算所需的数组。

# 相关报道
- [Guide To Qlib: Microsoft’s AI Investment Platform](https://analyticsindiamag.com/qlib/)
- [微软也搞AI量化平台？还是开源的！](https://mp.weixin.qq.com/s/47bP5YwxfTp2uTHjUBzJQQ)
- [微矿Qlib：业内首个AI量化投资开源平台](https://mp.weixin.qq.com/s/vsJv7lsgjEi-ALYUz4CvtQ)

# 联系我们
- 如有任何问题，请在 [这里](https://github.com/ssvip9527/qlib/issues/new/choose) 提 issue，或在 [gitter](https://gitter.im/Microsoft/qlib) 留言。
- 如希望为 `Qlib` 做贡献，请[提交 pull request](https://github.com/ssvip9527/qlib/compare)。
- 其他事宜欢迎邮件联系（[qlib@microsoft.com](mailto:qlib@microsoft.com)）。
  - 我们正在招聘新成员（全职/实习），欢迎投递简历！

加入 IM 讨论群：
|[Gitter](https://gitter.im/Microsoft/qlib)|
|----|
|![image](https://github.com/ssvip9527/qlib/blob/main/docs/_static/img/qrcode/gitter_qr.png)|

# 贡献
感谢所有贡献者！
<a href="https://github.com/ssvip9527/qlib/graphs/contributors"><img src="https://contrib.rocks/image?repo=microsoft/qlib" /></a>

Qlib 于 2020 年 9 月开源前为组内项目，遗憾的是内部提交历史未保留。组内许多成员也为 Qlib 做出了重要贡献，包括 Ruihua Wang、Yinda Zhang、Haisu Yu、Shuyu Wang、Bochen Pang 及 [Dong Zhou](https://github.com/evanzd/evanzd)。特别感谢 [Dong Zhou](https://github.com/evanzd/evanzd) 的初始版本。

## 指南

本项目欢迎各种贡献和建议。
**[代码规范与开发指南](docs/developer/code_standard_and_dev_guide.rst) 可参考提交 pull request。**

贡献并不难。解决 issue（如回答 [issues 列表](https://github.com/ssvip9527/qlib/issues) 或 [gitter](https://gitter.im/Microsoft/qlib) 上的问题）、修复/提出 bug、完善文档甚至修正错别字，都是对 Qlib 的重要贡献。

如想为 Qlib 文档/代码做贡献，可参考下图步骤：
<p align="center">
  <img src="https://github.com/demon143/qlib/blob/main/docs/_static/img/change%20doc.gif" />
</p>

如不知如何开始，可参考以下示例：
| 类型 | 示例 |
| -- | -- |
| 解决问题 | [回答问题](https://github.com/ssvip9527/qlib/issues/749)；[提出](https://github.com/ssvip9527/qlib/issues/765)或[修复](https://github.com/ssvip9527/qlib/pull/792) bug |
| 文档 | [提升文档质量](https://github.com/ssvip9527/qlib/pull/797/files)；[修正错别字](https://github.com/ssvip9527/qlib/pull/774) |
| 新特性 | [实现需求特性](https://github.com/ssvip9527/qlib/projects) 如 [此](https://github.com/ssvip9527/qlib/pull/754)；[重构接口](https://github.com/ssvip9527/qlib/pull/539/files) |
| 数据集 | [新增数据集](https://github.com/ssvip9527/qlib/pull/733) |
| 模型 | [实现新模型](https://github.com/ssvip9527/qlib/pull/689)，[贡献模型说明](https://github.com/ssvip9527/qlib/tree/main/examples/benchmarks#contributing) |

[Good first issues](https://github.com/ssvip9527/qlib/labels/good%20first%20issue) 标签表示适合新手入门。

可通过 `rg 'TODO|FIXME' qlib` 查找 Qlib 中待完善实现。

如希望成为 Qlib 维护者（如协助合并 PR、管理 issue），请邮件联系（[qlib@microsoft.com](mailto:qlib@microsoft.com)）。我们乐于协助提升权限。

## 许可证
大多数贡献需签署贡献者许可协议（CLA），声明你有权并实际授权我们使用你的贡献。详情见 https://cla.opensource.microsoft.com。

提交 pull request 时，CLA 机器人会自动判断是否需签署 CLA 并做相应标记（如状态检查、评论）。请按机器人指引操作。所有使用 CLA 的仓库只需签署一次。

本项目采用 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
更多信息见 [行为准则 FAQ](https://opensource.microsoft.com/codeofconduct/faq/)，或邮件联系 [opencode@microsoft.com](mailto:opencode@microsoft.com)。
