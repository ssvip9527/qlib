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

When you submit a PR request, you can check whether your code passes the CI tests in the "check" section at the bottom of the web page.

1. Qlib will check the code format with black. The PR will raise error if your code does not align to the standard of Qlib(e.g. a common error is the mixed use of space and tab).

   You can fix the bug by inputting the following code in the command line.

.. code-block:: bash

    pip install black
    python -m black . -l 120


2. Qlib will check your code style pylint. The checking command is implemented in [github action workflow](https://github.com/ssvip9527/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L66).
   Sometime pylint's restrictions are not that reasonable. You can ignore specific errors like this

.. code-block:: python

    return -ICLoss()(pred, target, index)  # pylint: disable=E1130


3. Qlib will check your code style flake8. The checking command is implemented in [github action workflow](https://github.com/ssvip9527/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L73).

   You can fix the bug by inputing the following code in the command line.

.. code-block:: bash

    flake8 --ignore E501,F541,E402,F401,W503,E741,E266,E203,E302,E731,E262,F523,F821,F811,F841,E713,E265,W291,E712,E722,W293 qlib


4. Qlib has integrated pre-commit, which will make it easier for developers to format their code.

   Just run the following two commands, and the code will be automatically formatted using black and flake8 when the git commit command is executed.

.. code-block:: bash

    pip install -e .[dev]
    pre-commit install


=================================
开发指南
=================================

作为开发者，您可能希望修改 `Qlib` 后无需重新安装就能直接在环境中生效。您可以通过以下命令以可编辑模式安装 `Qlib`。
The `[dev]` option will help you to install some related packages when developing `Qlib` (e.g. pytest, sphinx)

.. code-block:: bash

    pip install -e ".[dev]"