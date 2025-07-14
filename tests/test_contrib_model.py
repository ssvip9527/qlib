# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

from qlib.contrib.model import all_model_classes


class TestAllFlow(unittest.TestCase):
    def test_0_initialize(self):
        # 初始化计数器
        num = 0
        # 遍历所有模型类
        for model_class in all_model_classes:
            # 检查模型类是否有效
            if model_class is not None:
                # 实例化模型
                model = model_class()
                num += 1
        # 打印有效模型数量
        print("共有 {:}/{:} 个有效模型。".format(num, len(all_model_classes)))


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_initialize"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
