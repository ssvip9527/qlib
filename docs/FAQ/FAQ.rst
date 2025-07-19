
Qlib常见问题
############

Qlib常见问题解答
===============================
.. contents::
    :depth: 1
    :local:
    :backlinks: none

------


1. RuntimeError: 尝试在当前进程完成引导阶段之前启动新进程...
-----------------------------------------------------------------------------------------------------------------------------------

.. code-block:: console

    RuntimeError:
            尝试在当前进程完成引导阶段之前启动新进程。
            这可能意味着您没有使用 fork 来启动子进程，并且您忘记使用正确的语法。
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.

这是由于Windows操作系统下多进程的限制造成的。更多信息请参考 `这里 <https://stackoverflow.com/a/24374798>`_。

**解决方案**：在主模块的if __name__ == '__main__'子句中使用``D.features``来选择启动方法。例如：

.. code-block:: python

    import qlib
    from qlib.data import D


    if __name__ == "__main__":
        qlib.init()
        instruments = ["SH600000"]
        fields = ["$close", "$change"]
        df = D.features(instruments, fields, start_time='2010-01-01', end_time='2012-12-31')
        print(df.head())



2. qlib.data.cache.QlibCacheException: 发现Redis数据库中已存在Redis锁的key(...)。
---------------------------------------------------------------------------------------------------------------

发现Redis数据库中已存在Redis锁的key。您可以使用以下命令清除Redis键并重新运行命令

.. code-block:: console

    $ redis-cli
    > select 1
    > flushdb

如果问题未解决，使用``keys``* 查看是否存在多个键。如果存在，请尝试使用``flushall``清除所有键。

.. note::

    ``qlib.config.redis_task_db``的默认值为``1``，用户可以使用``qlib.init(redis_task_db=<other_db>)``进行设置。


另外，欢迎在我们的GitHub仓库中提交新的issue。我们会认真检查每个issue并尽力解决。

3. ModuleNotFoundError: 没有名为'qlib.data._libs.rolling'的模块
-----------------------------------------------------------------

.. code-block:: python

    #### 不要在仓库目录中导入qlib包，以避免未编译就从当前目录导入qlib #####
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "qlib/qlib/__init__.py", line 19, in init
        from .data.cache import H
    File "qlib/qlib/data/__init__.py", line 8, in <module>
        from .data import (
    File "qlib/qlib/data/data.py", line 20, in <module>
        from .cache import H
    File "qlib/qlib/data/cache.py", line 36, in <module>
        from .ops import Operators
    File "qlib/qlib/data/ops.py", line 19, in <module>
        from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    ModuleNotFoundError: No module named 'qlib.data._libs.rolling'

- 如果在使用``PyCharm`` IDE导入``qlib``包时出现此错误，用户可以在项目根目录执行以下命令编译Cython文件并生成可执行文件：

    .. code-block:: bash

        python setup.py build_ext --inplace

- 如果在使用``python``命令导入``qlib``包时出现此错误，用户需要更改运行目录，确保脚本不在项目目录中运行。


4. BadNamespaceError: /不是已连接的命名空间
----------------------------------------------------

.. code-block:: python

      File "qlib_online.py", line 35, in <module>
        cal = D.calendar()
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 973, in calendar
        return Cal.calendar(start_time, end_time, freq, future=future)
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 798, in calendar
        self.conn.send_request(
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\client.py", line 101, in send_request
        self.sio.emit(request_type + "_request", request_content)
      File "G:\apps\miniconda\envs\qlib\lib\site-packages\python_socketio-5.3.0-py3.8.egg\socketio\client.py", line 369, in emit
        raise exceptions.BadNamespaceError(
      BadNamespaceError: / is not a connected namespace.

- qlib中的``python-socketio``版本需要与qlib-server中的``python-socketio``版本相同：

    .. code-block:: bash

        pip install -U python-socketio==<qlib-server python-socketio version>


5. TypeError: send()收到意外的关键字参数'binary'
----------------------------------------------------------------

.. code-block:: python

      File "qlib_online.py", line 35, in <module>
        cal = D.calendar()
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 973, in calendar
        return Cal.calendar(start_time, end_time, freq, future=future)
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 798, in calendar
        self.conn.send_request(
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\client.py", line 101, in send_request
        self.sio.emit(request_type + "_request", request_content)
      File "G:\apps\miniconda\envs\qlib\lib\site-packages\socketio\client.py", line 263, in emit
        self._send_packet(packet.Packet(packet.EVENT, namespace=namespace,
      File "G:\apps\miniconda\envs\qlib\lib\site-packages\socketio\client.py", line 339, in _send_packet
        self.eio.send(ep, binary=binary)
      TypeError: send() got an unexpected keyword argument 'binary'


- 注意: ``python-engineio``版本需要与``python-socketio`` 版本兼容，参考：https://github.com/miguelgrinberg/python-socketio#version-compatibility

    .. code-block:: bash

        pip install -U python-engineio==<compatible python-socketio version>
        # or
        pip install -U python-socketio==3.1.2 python-engineio==3.13.2
