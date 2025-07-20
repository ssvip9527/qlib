.. _serial:

=============
序列化
=============
.. currentmodule:: qlib

简介
============
库``Qlib``支持将``DataHandler``、``DataSet``、``Processor``和``Model``等组件的状态转储到磁盘并重新加载它们。

可序列化类
==================

``Qlib``提供了一个基类``qlib.utils.serial.Serializable``，其状态可以以``pickle``格式转储到磁盘或从磁盘加载。
当用户转储``Serializable``实例的状态时，名称**不以**``_``开头的实例属性将被保存到磁盘。不过，用户可以使用``config``方法或重写``default_dump_all``属性来禁用此功能。

用户还可以重写``pickle_backend``属性来选择pickle后端。支持的值为``pickle``（默认且常用）和``dill``（可转储更多内容，如函数，更多信息见`这里 <https://pypi.org/project/dill/>`_）。

示例
=======
``Qlib``的可序列化类包括``DataHandler``、``DataSet``、``Processor``和``Model``等，它们都是``qlib.utils.serial.Serializable``的子类。
具体来说，``qlib.data.dataset.DatasetH``就是其中之一。用户可以按如下方式序列化``DatasetH``。

.. code-block:: Python

    ##=============dump dataset=============
    dataset.to_pickle(path="dataset.pkl") # dataset is an instance of qlib.data.dataset.DatasetH

    ##=============reload dataset=============
    with open("dataset.pkl", "rb") as file_dataset:
        dataset = pickle.load(file_dataset)

.. note::
    只有``DatasetH``的状态应该保存到磁盘，例如用于数据归一化的一些``mean``（均值）和``variance``（方差）等。

    重新加载``DatasetH``后，用户需要重新初始化它。这意味着用户可以重置``DatasetH``或``QlibDataHandler``的一些状态，如``instruments``（标的）、``start_time``（开始时间）、``end_time``（结束时间）和``segments``（分段）等，并根据这些状态生成新数据（数据不是状态，不应保存到磁盘）。

更详细的示例见此`链接 <https://github.com/ssvip9527/qlib/tree/main/examples/highfreq>`_。


API
===
请参考`Serializable API <../reference/api.html#module-qlib.utils.serial.Serializable>`_。