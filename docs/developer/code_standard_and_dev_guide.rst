.. _code_standard:

=============
代码规范
=============

文档字符串
=========
请使用 `Numpydoc 风格 <https://stackoverflow.com/a/24385103>`_。

持续集成
======================
持续集成（CI）工具通过在每次推送新提交时运行测试并向拉取请求报告结果，帮助您遵守质量标准。

当您提交PR请求时，可以在网页底部的"check"部分查看您的代码是否通过了CI测试。

1. Qlib将使用black检查代码格式。如果您的代码不符合Qlib的标准（例如，常见错误是空格和制表符混用），PR将会报错。

   您可以通过在命令行中输入以下代码来修复这个问题。

.. code-block:: bash

    pip install black
    python -m black . -l 120


2. Qlib将使用pylint检查您的代码风格。检查命令在[github action workflow](https://github.com/ssvip9527/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L66)中实现。
   有时pylint的限制并不那么合理。您可以像这样忽略特定错误

.. code-block:: python

    return -ICLoss()(pred, target, index)  # pylint: disable=E1130


3. Qlib将使用flake8检查您的代码风格。检查命令在[github action workflow](https://github.com/ssvip9527/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L73)中实现。

   您可以通过在命令行中输入以下代码来修复这个问题。

.. code-block:: bash

    flake8 --ignore E501,F541,E402,F401,W503,E741,E266,E203,E302,E731,E262,F523,F821,F811,F841,E713,E265,W291,E712,E722,W293 qlib


4. Qlib已集成了pre-commit，这将使开发者更容易格式化他们的代码。

   只需运行以下两个命令，当执行git commit命令时，代码将自动使用black和flake8进行格式化。

.. code-block:: bash

    pip install -e .[dev]
    pre-commit install


=================================
开发指南
=================================

作为开发者，您可能希望修改 `Qlib` 后无需重新安装就能直接在环境中生效。您可以通过以下命令以可编辑模式安装 `Qlib`。
`[dev]` 选项将帮助您在开发 `Qlib` 时安装一些相关的包（例如 pytest、sphinx）

.. code-block:: bash

    pip install -e ".[dev]"