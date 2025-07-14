# 众包数据源

## 项目背景
像雅虎这样的公共数据源存在缺陷，可能会遗漏已退市股票的数据，也可能包含错误数据。这会在我们的训练过程中引入幸存者偏差。

引入众包数据源旨在合并多个来源的数据并进行交叉验证，以实现：
1. 获得更完整的历史记录。
2. 能够识别异常数据并在必要时进行修正。

## 相关仓库
原始数据托管在dolthub仓库：https://www.dolthub.com/repositories/chenditc/investment_data

处理脚本和SQL文件托管在github仓库：https://github.com/chenditc/investment_data

打包的docker运行环境托管在dockerhub：https://hub.docker.com/repository/docker/chenditc/investment_data

## 在Qlib中使用方法
### 选项1：下载发布的二进制数据
用户可以下载qlib二进制格式的数据直接使用：https://github.com/chenditc/investment_data/releases/latest
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

### 选项2：从dolthub生成qlib数据
Dolthub数据每日更新，因此如果用户需要获取最新数据，可以使用docker导出qlib二进制文件：
```
docker run -v /<some output directory>:/output -it --rm chenditc/investment_data bash dump_qlib_bin.sh && cp ./qlib_bin.tar.gz /output/
```

## 常见问题及其他信息
详见：https://github.com/chenditc/investment_data/blob/main/README.md
