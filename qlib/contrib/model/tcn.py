# MIT License
# Copyright (c) 2018 CMU Locus Lab
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """用于移除卷积后的填充部分的模块

    参数:
        chomp_size: 需要移除的填充大小
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """前向传播: 移除张量末尾的填充部分

        参数:
            x: 输入张量

        返回:
            移除填充后的张量
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """时间卷积网络中的基本模块，包含两个卷积层和残差连接

    参数:
        n_inputs: 输入通道数
        n_outputs: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        dilation: 膨胀率
        padding: 填充大小
        dropout: Dropout概率
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """前向传播: 通过卷积块并应用残差连接

        参数:
            x: 输入张量

        返回:
            经过卷积块和残差连接后的输出
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """时间卷积网络，由多个TemporalBlock组成

    参数:
        num_inputs: 输入特征数
        num_channels: 每层输出通道数的列表
        kernel_size: 卷积核大小
        dropout: Dropout概率
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播: 通过时间卷积网络处理输入

        参数:
            x: 输入张量

        返回:
            经过网络处理后的输出
        """
        return self.network(x)
