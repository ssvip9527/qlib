.. _installation:

============
Installation
============

.. currentmodule:: qlib


``Qlib`` Installation
=====================
.. note::

   `Qlib支持`Windows`和`Linux`系统。建议在`Linux`环境中使用Qlib。Qlib支持Python3，最高兼容Python3.8版本。

用户可以通过以下命令使用pip轻松安装Qlib：

.. code-block:: bash

   pip install pyqlib


此外，用户还可以按照以下步骤通过源代码安装Qlib：

- 进入Qlib的根目录，该目录下应包含``setup.py``文件。
- 然后执行以下命令安装环境依赖并安装Qlib：

   .. code-block:: bash

      $ pip install numpy
      $ pip install --upgrade cython
      $ git clone https://github.com/microsoft/qlib.git && cd qlib
      $ python setup.py install

.. note::
   建议使用anaconda/miniconda来设置环境。Qlib需要lightgbm和pytorch包，请使用pip安装它们。



使用以下代码验证安装是否成功：

.. code-block:: python

   >>> import qlib
   >>> qlib.__version__
   <LATEST VERSION>
