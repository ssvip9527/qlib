import unittest
from qlib.tests import TestAutoData


class TestNN(TestAutoData):
    def test_both_dataset(self):
        try:
            from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
            from qlib.data.dataset import DatasetH, TSDatasetH
            from qlib.data.dataset.handler import DataHandlerLP
        except ImportError:
            print("Import error.")
            return

        data_handler_config = {
            "start_time": "2008-01-01",
            "end_time": "2020-08-01",
            "instruments": "csi300",
            "data_loader": {
                "class": "QlibDataLoader",  # 假设QlibDataLoader是类的字符串引用
                "kwargs": {
                    "config": {
                        "feature": [["$high", "$close", "$low"], ["H", "C", "L"]],
                        "label": [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]],
                    },
                    "freq": "day",
                },
            },
            # TODO: 处理器
            "learn_processors": [
                {
                    "class": "DropnaLabel",
                },
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
        }
        segments = {
            "train": ["2008-01-01", "2014-12-31"],
            "valid": ["2015-01-01", "2016-12-31"],
            "test": ["2017-01-01", "2020-08-01"],
        }
        data_handler = DataHandlerLP(**data_handler_config)

        # 时间序列数据集
        tsds = TSDatasetH(handler=data_handler, segments=segments)

        # 表格数据集
        tbds = DatasetH(handler=data_handler, segments=segments)

        model_l = [
            GeneralPTNN(
                n_epochs=2,
                batch_size=32,
                n_jobs=0,
                pt_model_uri="qlib.contrib.model.pytorch_gru_ts.GRUModel",
                pt_model_kwargs={
                    "d_feat": 3,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "dropout": 0.0,
                },
            ),
            GeneralPTNN(
                n_epochs=2,
                batch_size=32,
                n_jobs=0,
                pt_model_uri="qlib.contrib.model.pytorch_nn.Net",  # 这是一个MLP
                pt_model_kwargs={
                    "input_dim": 3,
                },
            ),
        ]

        for ds, model in list(zip((tsds, tbds), model_l)):
            model.fit(ds)  # 可以正常工作
            model.predict(ds)  # 可以正常工作


if __name__ == "__main__":
    unittest.main()
