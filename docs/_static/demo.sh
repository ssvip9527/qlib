#!/bin/sh
git clone https://github.com/ssvip9527/qlib.git
cd qlib
ls
pip install pyqlib
# 或者
# pip install numpy
# pip install --upgrade cython
# python setup.py install
cd examples
ls
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml