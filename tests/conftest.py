import os
import sys

"""在非 Linux 平台上忽略 RL 测试。"""
collect_ignore = []

if sys.platform != "linux":
    for root, dirs, files in os.walk("rl"):
        for file in files:
            collect_ignore.append(os.path.join(root, file))
