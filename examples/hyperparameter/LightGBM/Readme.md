# LightGBM超参数

## Alpha158
第一个终端
```
optuna create-study --study LGBM_158 --storage sqlite:///db.sqlite3
optuna-dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```
第二个终端
```
python hyperparameter_158.py
```

## Alpha360
第一个终端
```
optuna create-study --study LGBM_360 --storage sqlite:///db.sqlite3
optuna-dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```
第二个终端
```
python hyperparameter_360.py
```
