# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import platform
import mlflow
import time
from pathlib import Path
import shutil


class MLflowTest(unittest.TestCase):
    TMP_PATH = Path("./.mlruns_tmp/")

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_creating_client(self):
        """
        请参考qlib/workflow/expm.py:MLflowExpManager._client
        我们不缓存_client（这有助于在MLflowExpManager的uri更改时减少维护工作）

        此实现基于创建客户端速度很快的假设
        """
        start = time.time()
        for i in range(10):
            _ = mlflow.tracking.MlflowClient(tracking_uri=str(self.TMP_PATH))
        end = time.time()
        elapsed = end - start
        if platform.system() == "Linux":
            self.assertLess(elapsed, 1e-2)  # 可以在10毫秒内完成
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)


if __name__ == "__main__":
    unittest.main()
