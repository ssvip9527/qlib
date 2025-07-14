.. _server:

=============================
``在线`` & ``离线`` 模式
=============================
.. currentmodule:: qlib


简介
============

``Qlib``支持``在线``模式和``离线``模式。本文档仅介绍``离线``模式。

``在线``模式旨在解决以下问题：

- 集中管理数据。用户无需管理不同版本的数据。
- 减少需要生成的缓存量。
- 支持远程访问数据。

Qlib-Server
===========

``Qlib-Server``是``Qlib``的配套服务器系统，它利用``Qlib``进行基础计算，并提供完善的服务器系统和缓存机制。通过Qlib-Server，可以集中管理提供给``Qlib``的数据。使用``Qlib-Server``，用户可以在``在线``模式下使用``Qlib``。



参考
=========
如果用户对``Qlib-Server``和``在线``模式感兴趣，请参考`Qlib-Server项目 <https://github.com/microsoft/qlib-server>`_和`Qlib-Server文档 <https://qlib-server.readthedocs.io/en/latest/>`_。
