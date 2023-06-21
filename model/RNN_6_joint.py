from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

import torch.nn as nn
import torch
from torch import tanh
from torch.nn.parameter import Parameter
import math

"""
目前删除pro和GCN模块，仅保留RNN
"""


class Predict_imu(nn.Module):
    def __init__(self,
                 input_frame_len,
                 output_frame_len,
                 input_size,  # 120
                 mid_size,
                 output_size,  # 30
                 batch_size,
                 adjs,
                 device,
                 dropout,
                 ):
        super(Predict_imu, self).__init__()
        """
        Args:
            input_frame_len:输入帧数
            output_frame_len:输出帧数
            input_size:输入维度72
            out_size:输出维度18
            batch_size:batch 可以考虑使用for循环
            adjs:邻接矩阵
            num_layers:
            device:cuda or cpu
            dropout:dropout几率
        """
        # 训练的参数
        self.input_frame_len = input_frame_len
        self.output_frame_len = output_frame_len
        self.input_size = input_size
        self.mid_size = mid_size
        self.output_size = output_size
        self.adjs = adjs
        self.device = device
        self.dropout = dropout
        # 网络结构
        hidden_size_pos = 256

        self.rnn_pos = RNN_pos(frame_num=input_frame_len,
                               input_size=input_size,  # 需要修改
                               output_size=output_size,
                               hidden_size=256,
                               num_layers=2,
                               dropout=dropout
                               )
        # self.gcn1 = GC_Block(in_features=input_frame_len, p_dropout=dropout, bias=True, node_n=mid_size)

        # 输出frame,batch,dim=18
        self.all_position = RNN_pos_all(frame_num=input_frame_len,
                                        input_size=input_size,  # 需要修改
                                        output_size=output_size,
                                        hidden_size=64,
                                        num_layers=3,
                                        dropout=dropout)
        self.gcn2 = GC_Block(in_features=input_frame_len, p_dropout=dropout, bias=True, node_n=output_size)

    def forward(self, ori, acc):
        acc = torch.transpose(acc, 0, 1)
        ori = torch.transpose(ori, 0, 1)
        f, b, _, _ = acc.shape
        posi_24 = self.rnn_pos(ori, acc).reshape(f, b, 24, -1)
        # y = self.gcn1(posi_6)
        # posi_24 = self.gcn2(posi_24)
        return torch.transpose(posi_24, 0, 1)


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    # 初始化
    # 输入参数，输出参数，权重，偏移 node_n是节点数量
    def __init__(self, in_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()  # 不加super会报错
        self.in_features = in_features
        '''
        将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的)，
        '''
        self.weight = Parameter(torch.FloatTensor(in_features, in_features).to(torch.float32))  # 权重设置为parameter类
        self.att = Parameter(torch.FloatTensor(node_n, node_n).to(torch.float32))  # 可学习的加权邻接矩阵
        # 每个图卷积层都有一个不同的可学习的加权邻接矩阵A，可以使网络适应不同操作的连通性
        if bias:
            self.bias = Parameter(torch.FloatTensor(node_n))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数初始化
    # 初始化权重，均匀分布随机生成在-stdv到stdv之间的数字
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向函数
    def forward(self, input):
        support = torch.matmul(input, self.att)  # 高维矩阵相乘
        output = torch.matmul(self.weight, support)  # 三个矩阵相乘input X与权重W相乘，然后adj矩阵与 他们的积稀疏乘。(AHW)三个举证相乘
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output

    # 定义格式
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias, node_n):
        """
        Define a residual block of GCN定义GCN的剩余块
        """
        super(GC_Block, self).__init__()  # 没有super函数会报错
        self.in_features = in_features
        self.out_features = in_features
        # 下面两个有什么区别呢？
        self.gc1 = GraphConvolution(in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.gc2 = GraphConvolution(in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)  # 输入参数是前一层输出的维度
        self.do = nn.Dropout(p_dropout)  # dropout层
        self.act_f = nn.Tanh()  # 激励函数双曲正切函数

    def forward(self, x):
        f, b, n = x.shape
        y = x.transpose(0, 1)
        y = self.gc1(y)
        y = self.bn1(y.view(b, -1)).view(b, f, n)
        y = self.act_f(y)
        y = self.do(y)
        '''
        gc1 GCN卷积核
        bn1 归一化层
        act_f 激励函数
        do dropout层
        '''
        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, f, n)  # (b,-1)表示分成b行，平均分，然后每一行再分成n行f列
        y = self.act_f(y)
        y = self.do(y).transpose(0, 1)
        '''
        gc2 GCN卷积核
        bn2 归一化层
        act_f 激励函数
        do dropout层              
        '''
        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RNN_pos(nn.Module):
    def __init__(self,
                 frame_num,  # 帧数
                 input_size,  # 输入格式
                 output_size,  # 输出格式
                 hidden_size,
                 num_layers,
                 dropout):
        super(RNN_pos, self).__init__()
        self.frame_num = frame_num
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        # 网络结构
        self.bilstm = BiLSTM(input_size=input_size,
                             output_size=output_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)

    def forward(self, ori, acc):
        encoder_inputs = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2)
        y = self.bilstm(encoder_inputs)
        return y


class RNN_pos_all(nn.Module):
    def __init__(self,
                 frame_num,  # 帧数
                 input_size,  # 输入格式
                 output_size,  # 输出格式
                 hidden_size,
                 num_layers,
                 dropout):
        super(RNN_pos_all, self).__init__()
        self.frame_num = frame_num
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        # 网络结构
        self.bilstm = BiLSTM(input_size=input_size,
                             output_size=output_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)

    def forward(self, ori, acc):
        encoder_inputs = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2)
        f, b, _ = encoder_inputs.shape
        y = self.bilstm(encoder_inputs)
        return y


class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(BiLSTM, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.bi_lstm = torch.nn.LSTM(hidden_size, hidden_size,
                                     bidirectional=True,
                                     num_layers=num_layers
                                     )
        self.fc2 = torch.nn.Linear(in_features=hidden_size * 2, out_features=output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, hid=None):
        # 开始进行RNN传播
        output, hid = self.bi_lstm(tanh(self.fc1(self.dropout(x))), hid)
        output = self.fc2(output)
        return output
