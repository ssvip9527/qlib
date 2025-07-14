# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from pathlib import Path
import shutil

from qlib.workflow import R
from qlib.tests import TestAutoData


class WorkflowTest(TestAutoData):
    # 手动创建目录对 mlflow 无效，
    # 因此我们在创建目录时添加一个名为 .trash 的子文件夹。
    TMP_PATH = Path("./.mlruns_tmp/.trash")

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_get_local_dir(self):
        """测试获取本地目录功能"""
        self.TMP_PATH.mkdir(parents=True, exist_ok=True)

        with R.start(uri=str(self.TMP_PATH)):
            pass

        with R.uri_context(uri=str(self.TMP_PATH)):
            resume_recorder = R.get_recorder()
            resume_recorder.get_local_dir()


if __name__ == "__main__":
    unittest.main()
