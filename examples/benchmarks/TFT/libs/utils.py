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
"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """返回单个列的名称。

    参数:
      input_type: 要提取的列的输入类型
      column_definition: 实验的列定义列表
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

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


# Loss functions.
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

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError("Illegal quantile value={}! Values should be between 0 and 1.".format(quantile))

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


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

    Args:
      directory: Folder path to create.
    """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device="gpu", gpu_id=0):
    """Creates tensorflow config for graphs to run on CPU or GPU.

    Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
    GPU machines.

    Args:
      tf_device: 'cpu' or 'gpu'
      gpu_id: GPU ID to use if relevant

    Returns:
      Tensorflow config.
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
    """Saves Tensorflow graph to checkpoint.

    Saves all trainiable variables under a given variable scope to checkpoint.

    Args:
      tf_session: Session containing graph
      model_folder: Folder to save models
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope containing variables to save
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
    """Loads Tensorflow graph from checkpoint.

    Args:
      tf_session: Session to load graph into
      model_folder: Folder containing serialised model
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope to use.
      verbose: Whether to print additional debugging information.
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
