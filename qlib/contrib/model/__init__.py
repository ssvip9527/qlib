# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
try:
    from .catboost_model import CatBoostModel
except ModuleNotFoundError:
    CatBoostModel = None
    print("ModuleNotFoundError. CatBoostModel已被跳过。（可选：安装CatBoostModel可能可以解决此问题。）")
try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print(
        "ModuleNotFoundError. DEnsembleModel和LGBModel已被跳过。（可选：安装lightgbm可能可以解决此问题。）"
    )
try:
    from .xgboost import XGBModel
except ModuleNotFoundError:
    XGBModel = None
    print("ModuleNotFoundError. XGBModel已被跳过（可选：安装xgboost可能可以解决此问题）。")
try:
    from .linear import LinearModel
except ModuleNotFoundError:
    LinearModel = None
    print("ModuleNotFoundError. LinearModel已被跳过（可选：安装scipy和sklearn可能可以解决此问题）。")
# import pytorch models
try:
    from .pytorch_alstm import ALSTM
    from .pytorch_gats import GATs
    from .pytorch_gru import GRU
    from .pytorch_lstm import LSTM
    from .pytorch_nn import DNNModelPytorch
    from .pytorch_tabnet import TabnetModel
    from .pytorch_sfm import SFM_Model
    from .pytorch_tcn import TCN
    from .pytorch_add import ADD

    pytorch_classes = (ALSTM, GATs, GRU, LSTM, DNNModelPytorch, TabnetModel, SFM_Model, TCN, ADD)
except ModuleNotFoundError:
    pytorch_classes = ()
    print("ModuleNotFoundError.  PyTorch模型已被跳过（可选：安装pytorch可能可以解决此问题）。")

all_model_classes = (CatBoostModel, DEnsembleModel, LGBModel, XGBModel, LinearModel) + pytorch_classes
