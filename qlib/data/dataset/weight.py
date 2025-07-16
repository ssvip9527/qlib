# 版权所有 (c) Microsoft Corporation。
# 基于 MIT 许可证授权。


class Reweighter:
    def __init__(self, *args, **kwargs):
        """
        要初始化Reweighter，用户应提供特定方法让重加权器执行重加权操作（如样本级、规则基）。
        """
        raise NotImplementedError()

    def reweight(self, data: object) -> object:
        """
        为数据获取权重

        参数
        ----------
        data : object
            输入数据。
            第一维度是样本索引

        返回
        -------
        object:
            数据的权重信息
        """
        raise NotImplementedError(f"This type of input is not supported")
