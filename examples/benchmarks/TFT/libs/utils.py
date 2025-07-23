# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""整个代码库中使用的通用辅助函数。"""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# 通用函数
def get_single_col_by_input_type(input_type, column_definition):
    """返回单个列的名称。

    参数:
      input_type: 要提取的列的输入类型
      column_definition: 实验的列定义列表
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("{}的列数无效".format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """提取与指定数据类型对应列的名称。

    参数:
      data_type: 要提取的列的数据类型。
      column_definition: 要使用的列定义。
      excluded_input_types: 要排除的输入类型集合

    返回:
      指定数据类型的列名称列表。
    """
    return [tup[0] for tup in column_definition if tup[1] == data_type and tup[2] not in excluded_input_types]


# 损失函数
def tensorflow_quantile_loss(y, y_pred, quantile):
    """为TensorFlow计算分位数损失。

    标准分位数损失，定义于TFT主论文的"训练过程"部分

    参数:
      y: 目标值
      y_pred: 预测值
      quantile: 用于损失计算的分位数（介于0和1之间）

    返回:
      分位数损失的张量。
    """

    # 检查分位数
    if quantile < 0 or quantile > 1:
        raise ValueError("分位数值={}无效！值应该在0和1之间。".format(quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * tf.maximum(
        -prediction_underflow, 0.0
    )

    return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    """为numpy数组计算归一化分位数损失。

    使用TFT主论文"训练过程"部分定义的q-Risk指标。

    参数:
      y: 目标值
      y_pred: 预测值
      quantile: 用于损失计算的分位数（介于0和1之间）

    返回:
      归一化分位数损失的浮点值。
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(
        -prediction_underflow, 0.0
    )

    quantile_loss = weighted_errors.mean()
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


# 操作系统相关函数
def create_folder_if_not_exist(directory):
    """如果文件夹不存在则创建。

    参数:
      directory: 要创建的文件夹路径。
    """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# TensorFlow相关函数
def get_default_tensorflow_config(tf_device="gpu", gpu_id=0):
    """创建用于在CPU或GPU上运行图的TensorFlow配置。

    指定是在gpu还是cpu上运行图，以及在多GPU机器上使用哪个GPU ID。

    参数:
      tf_device: 'cpu'或'gpu'
      gpu_id: 要使用的GPU ID（如果相关）

    返回:
      TensorFlow配置。
    """

    if tf_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # for training on cpu
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print("Selecting GPU ID={}".format(gpu_id))

        tf_config = tf.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    """将TensorFlow图保存到检查点。

    将给定变量作用域下的所有可训练变量保存到检查点。

    参数:
      tf_session: 包含图的会话
      model_folder: 保存模型的文件夹
      cp_name: TensorFlow检查点的名称
      scope: 包含要保存变量的变量作用域
    """
    # Save model
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session, os.path.join(model_folder, "{0}.ckpt".format(cp_name)))
    print("Model saved to: {0}".format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """从检查点加载TensorFlow图。

    参数:
      tf_session: 要加载图的会话
      model_folder: 包含序列化模型的文件夹
      cp_name: TensorFlow检查点的名称
      scope: 要使用的变量作用域
      verbose: 是否打印额外的调试信息
    """
    # Load model proper
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print("Loading model from {0}".format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    if verbose:
        print("Restored {0}".format(",".join(initial_vars.difference(all_vars))))
        print("Existing {0}".format(",".join(all_vars.difference(initial_vars))))
        print("All {0}".format(",".join(all_vars)))

    print("Done.")


def print_weights_in_checkpoint(model_folder, cp_name):
    """Prints all weights in Tensorflow checkpoint.

    Args:
      model_folder: Folder containing checkpoint
      cp_name: Name of checkpoint

    Returns:

    """
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print_tensors_in_checkpoint_file(file_name=load_path, tensor_name="", all_tensors=True, all_tensor_names=True)
