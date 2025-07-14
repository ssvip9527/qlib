.. _docker_image:

==================
构建Docker镜像
==================

Dockerfile文件
==========

项目根目录下有一个**Dockerfile**文件，您可以通过该文件构建docker镜像。Dockerfile中有两种构建方式可供选择。
执行构建命令时，使用``--build-arg``参数控制镜像版本。``--build-arg``参数默认为``yes``，将构建qlib镜像的``stable``（稳定）版本。

1. 对于``stable``（稳定）版本，使用``pip install pyqlib``命令构建qlib镜像。

.. code-block:: bash

    docker build --build-arg IS_STABLE=yes -t <image name> -f ./Dockerfile .

.. code-block:: bash

    docker build -t <image name> -f ./Dockerfile .

2. 对于``nightly``（每日更新）版本，使用当前源代码构建qlib镜像。

.. code-block:: bash

    docker build --build-arg IS_STABLE=no -t <image name> -f ./Dockerfile .

qlib镜像的自动构建
=========================

1. 项目根目录下有一个**build_docker_image.sh**脚本文件，可用于自动构建docker镜像并上传到您的docker hub仓库（可选，需配置）。

.. code-block:: bash

    sh build_docker_image.sh
    >>> 是否构建qlib镜像的nightly版本？（默认为stable版本）(yes/no)：
    >>> 是否上传到docker hub？（默认为否）(yes/no)：

2. 如果您想将构建好的镜像上传到docker hub仓库，需要先编辑**build_docker_image.sh**文件，填写文件中的``docker_user``变量，然后执行该文件。

如何使用qlib镜像
======================
1. 启动新的Docker容器

.. code-block:: bash

    docker run -it --name <container name> -v <Mounted local directory>:/app <image name>

2. 此时您已进入docker环境，可以运行qlib脚本。示例：

.. code-block:: bash

    >>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
    >>> python qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

3. 退出容器

.. code-block:: bash

    >>> exit

4. 重启容器

.. code-block:: bash

    docker start -i -a <container name>

5. 停止容器

.. code-block:: bash

    docker stop -i -a <container name>

6. 删除容器

.. code-block:: bash

    docker rm <container name>

7. 有关docker使用的更多信息，请参阅`docker官方文档 <https://docs.docker.com/reference/cli/docker/>`_。
