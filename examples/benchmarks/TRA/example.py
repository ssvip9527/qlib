import argparse

import qlib
from ruamel.yaml import YAML
from qlib.utils import init_instance_by_config


def main(seed, config_file="configs/config_alstm.yaml"):
    # 设置随机种子
    with open(config_file) as f:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(f)

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # 初始化工作流
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    # 训练模型
    model.fit(dataset)


if __name__ == "__main__":
    # 从命令行设置参数
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="随机种子")
    parser.add_argument("--config_file", type=str, default="configs/config_alstm.yaml", help="配置文件")
    args = parser.parse_args()
    main(**vars(args))
