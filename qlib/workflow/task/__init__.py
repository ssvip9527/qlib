# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
任务相关的工作流程在此文件夹中实现

一个典型的工作流程

| 步骤                  | 描述                                    |
|-----------------------+------------------------------------------------|
| TaskGen               | 生成任务                              |
| TaskManager(可选) | 管理生成的任务                         |
| run task              | 从TaskManager检索任务并运行任务 |
"""
