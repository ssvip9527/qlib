# 版权所有 (c) 微软公司。
# 根据 MIT 许可证授权。
import os
from pathlib import Path
from typing import Union

import fire

from qlib import auto_init
from qlib.contrib.rolling.ddgda import DDGDA
from qlib.tests.data import GetData

DIRNAME = Path(__file__).absolute().resolve().parent
BENCH_DIR = DIRNAME.parent / "baseline"


class DDGDABench(DDGDA):
    # README.md 中的配置文件
    CONF_LIST = [
        BENCH_DIR / "workflow_config_linear_Alpha158.yaml",
        BENCH_DIR / "workflow_config_lightgbm_Alpha158.yaml",
    ]

    DEFAULT_CONF = CONF_LIST[0]  # 默认使用线性模型以提高效率

    def __init__(self, conf_path: Union[str, Path] = DEFAULT_CONF, horizon=20, **kwargs) -> None:
        # 此代码用于兼容早期旧版本代码
        conf_path = Path(conf_path)
        super().__init__(conf_path=conf_path, horizon=horizon, working_dir=DIRNAME, **kwargs)

        for f in self.CONF_LIST:
            if conf_path.samefile(f):
                break
        else:
            self.logger.warning("模型类型不在基准测试范围内！")


if __name__ == "__main__":
    kwargs = {}
    if os.environ.get("PROVIDER_URI", "") == "":
        GetData().qlib_data(exists_skip=True)
    else:
        kwargs["provider_uri"] = os.environ["PROVIDER_URI"]
    auto_init(**kwargs)
    fire.Fire(DDGDABench)
