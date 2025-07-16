# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from typing import Text, Union
from ..utils.serial import Serializable
from ..data.dataset import Dataset
from ..data.dataset.weight import Reweighter


class BaseModel(Serializable, metaclass=abc.ABCMeta):
    """基础模型类"""

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """模型训练完成后进行预测"""

    def __call__(self, *args, **kwargs) -> object:
        """利用Python语法糖使模型行为像函数一样"""
        return self.predict(*args, **kwargs)


class Model(BaseModel):
    """可学习模型"""

    def fit(self, dataset: Dataset, reweighter: Reweighter):
        """
        从基础模型学习模型

        .. note::

            学习模型的属性名称不应以'_'开头，以便模型可以序列化到磁盘。

        以下代码示例展示如何从dataset获取`x_train`、`y_train`和`w_train`:

            .. code-block:: Python

                # 获取特征和标签
                df_train, df_valid = dataset.prepare(
                    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                x_train, y_train = df_train["feature"], df_train["label"]
                x_valid, y_valid = df_valid["feature"], df_valid["label"]

                # 获取权重
                try:
                    wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"],
                                                           data_key=DataHandlerLP.DK_L)
                    w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
                except KeyError as e:
                    w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
                    w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

        参数
        ----------
        dataset : Dataset
            数据集将生成模型训练所需的处理后的数据

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, dataset: Dataset, segment: Union[Text, slice] = "test") -> object:
        """根据数据集进行预测

        参数
        ----------
        dataset : Dataset
            数据集将生成模型训练所需的处理后的数据

        segment : Text or slice
            数据集将使用该片段准备数据 (默认="test")

        返回
        -------
        预测结果，类型可能为`pandas.Series`等
        """
        raise NotImplementedError()


class ModelFT(Model):
    """可微调模型"""

    @abc.abstractmethod
    def finetune(self, dataset: Dataset):
        """基于给定数据集微调模型

        使用qlib.workflow.R微调模型的典型用例:

        .. code-block:: python

            # 开始实验训练初始模型
            with R.start(experiment_name="init models"):
                model.fit(dataset)
                R.save_objects(init_model=model)
                rid = R.get_recorder().id

            # 基于之前训练的模型进行微调
            with R.start(experiment_name="finetune model"):
                recorder = R.get_recorder(recorder_id=rid, experiment_name="init models")
                model = recorder.load_object("init_model")
                model.finetune(dataset, num_boost_round=10)


        参数
        ----------
        dataset : Dataset
            数据集将生成模型训练所需的处理后的数据
        """
        raise NotImplementedError()
