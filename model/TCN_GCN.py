from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np


class GraphConvolution(nn.Module):
    def __init__(self, filters, input_dim,
                 use_bias=True):
        super(GraphConvolution, self).__init__()
        self.units = filters
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(input_dim, self.units).to(torch.float32))  # 权重设置为parameter类
        # 每个图卷积层都有一个不同的可学习的加权邻接矩阵A，可以使网络适应不同操作的连通性
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(self.units))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # 参数初始化
        # 初始化权重，均匀分布随机生成在-stdv到stdv之间的数字

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, A):
        shape = features.shape
        if len(shape) > 3:
            features = features.transpose(1, 2)
            features = features.reshape(-1, features.shape[1], features.shape[2] * features.shape[3])

            output = torch.matmul(A, features)
            output = torch.matmul(output, self.weight)
        else:
            output = torch.matmul(A, features)
            output = torch.matmul(output, self.weight)

        if self.use_bias:
            output += self.bias
        return output


class Deep_Priors(nn.Module):
    def __init__(self, input_length, node_n, Adj, device, activation='relu', p_dropout=0.5, net_depth=4):
        super(Deep_Priors, self).__init__()
        self.motion_length = input_length
        self.node_n = node_n
        self.end_channel = self.motion_length * 3
        Q = Parameter(torch.FloatTensor(self.node_n, self.node_n))
        Q.data.uniform_(0.01, 0.24)
        self.Adj = (Adj + Q).to(device)
        self.activation = activation
        filters = [1024, 512, 256, 128, 128]
        self.E0 = GraphTemporalConvolution(
            filters=filters[0],
            input_dim=self.end_channel,
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.E1 = GraphTemporalConvolution(
            filters=filters[1],
            input_dim=filters[0],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.E2 = GraphTemporalConvolution(
            filters=filters[2],
            input_dim=filters[1],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.E3 = GraphTemporalConvolution(
            filters=filters[3],
            input_dim=filters[2],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.ED = GraphTemporalConvolution(
            filters=filters[4],
            input_dim=filters[3],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.D3 = GraphTemporalConvolution(
            filters=filters[3],
            input_dim=filters[4],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.D2 = GraphTemporalConvolution(
            filters=filters[2],
            input_dim=filters[3],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.D1 = GraphTemporalConvolution(
            filters=filters[1],
            input_dim=filters[2],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.D0 = GraphTemporalConvolution(
            filters=filters[0],
            input_dim=filters[1],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')
        self.DD = GraphTemporalConvolution(
            filters=self.end_channel,
            input_dim=filters[0],
            node_n=self.node_n,
            p_dropout=p_dropout, adj=self.Adj, act='mish')

    def forward(self, ori, acc):
        A = self.Adj
        encoder_input = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2)
        b, f, dim = encoder_input.shape
        encoder_input = encoder_input.reshape(b, f, self.node_n, -1)
        x0 = self.E0(encoder_input, A)
        x1 = self.E1(x0, A)
        x2 = self.E2(x1, A)
        x3 = self.E3(x2, A)
        xy = self.ED(x3, A)
        y3 = self.D3(xy, A) + x3
        y2 = self.D2(y3, A) + x2
        y1 = self.D1(y2, A) + x1
        y0 = self.D0(y1, A) + x0
        y = self.DD(y0, A)
        output = y.reshape(b, self.node_n, f, 3).transpose(1, 2)
        return output


def act_result(x, name='relu'):
    # using the optimal activation function
    if name == 'relu':
        act = nn.ReLU()
        return act(x)
    elif name == 'mish':
        act = nn.Mish()
        return act(x)
    elif name == 'gelu':
        act = nn.GELU()
        return act(x)
    elif name == 'swish':
        act = nn.Hardswish()
        return act(x)
    elif name == 'tanh':
        act = nn.Tanh()
        return act(x)
    else:
        return x


class GraphTemporalConvolution(nn.Module):
    def __init__(self, filters=1024, input_dim=180, use_bias=True, node_n=24, p_dropout=0.5, adj=None, resi=True,
                 T_kernel_size=9,
                 name='0', act='relu'):
        super(GraphTemporalConvolution, self).__init__()
        self.gc = GraphConvolution(filters=filters, input_dim=input_dim, use_bias=use_bias)
        self.bn1 = nn.BatchNorm1d(node_n)
        self.act = act
        self.res = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=9, padding='same')
        self.bn2 = nn.BatchNorm1d(node_n)
        self.resi = resi
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input, A):
        output = self.gc(input, A)
        y = self.bn1(output)
        y = act_result(y, self.act)
        y = self.dropout(y).transpose(1, 2)
        y = self.res(y).transpose(1, 2)
        y = self.bn2(y)
        if self.resi:
            return y + output
        else:
            return y
