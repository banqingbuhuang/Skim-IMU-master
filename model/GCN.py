#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

'''
如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，
也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入。
加上这些，如果你的python版本是python2.X，你也得按照python3.X那样使用这些函数。
'''
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


# 图卷积
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    # 初始化
    # 输入参数，输出参数，权重，偏移 node_n是节点数量
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()  # 不加super会报错
        self.in_features = in_features
        self.out_features = out_features
        '''
        将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的)，
        '''
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(torch.float32))  # 权重设置为parameter类
        self.att = Parameter(torch.FloatTensor(node_n, node_n).to(torch.float32))  # 可学习的加权邻接矩阵
        # 每个图卷积层都有一个不同的可学习的加权邻接矩阵A，可以使网络适应不同操作的连通性
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # 偏移为输出参数数量
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
    def forward(self, x):
        y = torch.matmul(x, self.weight)  # 高维矩阵相乘
        y = torch.matmul(self.att, y)  # 三个矩阵相乘input X与权重W相乘，然后adj矩阵与 他们的积稀疏乘。(AHW)三个举证相乘
        if self.bias is not None:
            return y + self.bias  # 如果有偏移就加上偏移值
        else:
            return y

    # 定义格式
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    # in_feature=256
    """
    GCN(
          (gc1): GraphConvolution (35 -> 256)输入35，输出256
          (bn1): BatchNorm1d(12288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)256*48=12288
          (gcbs): ModuleList(
            (0): GC_Block (256 -> 256)
            (1): GC_Block (256 -> 256)
            (2): GC_Block (256 -> 256)
            (3): GC_Block (256 -> 256)
            (4): GC_Block (256 -> 256)
            (5): GC_Block (256 -> 256)
            (6): GC_Block (256 -> 256)
            (7): GC_Block (256 -> 256)
            (8): GC_Block (256 -> 256)
            (9): GC_Block (256 -> 256)
            (10): GC_Block (256 -> 256)
            (11): GC_Block (256 -> 256)
          )
          (gc7): GraphConvolution (256 -> 35)
          (do): Dropout(p=0.5, inplace=False)
          (act_f): Tanh()
     )
     """


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN定义GCN的剩余块
        """
        super(GC_Block, self).__init__()  # 没有super函数会报错
        self.in_features = in_features
        self.out_features = in_features
        # 下面两个有什么区别呢？
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        '''
        深层网络所需要的 在深度网络的中间层内添加正态标准化处理 可以防止出现梯度爆炸/消失问题
         通过归一化神经元的输出，激活函数将仅接收接近零的输入。
         所以使用tanh()作为激活函数，却不用担心梯度消失的问题
         通过批标准化，我们确保任何激活函数的输入不会进入饱和区域。批量归一化将这些输入的分布转换为0-1高斯分布。
         通过防止在训练期间消失梯度的问题，我们可以设置更高的学习率。批量标准化还降低了对参数标度的依赖性。
         大的学习速率可以增加层参数的规模，这导致梯度在反向传播期间被回传时放大。
        '''
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)  # 输入参数是前一层输出的维度
        self.do = nn.Dropout(p_dropout)  # dropout层
        self.act_f = nn.Tanh()  # 激励函数双曲正切函数

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        '''
        gc1 GCN卷积核
        bn1 归一化层
        act_f 激励函数
        do dropout层
        '''
        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)  # (b,-1)表示分成b行，平均分，然后每一行再分成n行f列
        y = self.act_f(y)
        y = self.do(y)
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


# input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,num_stage=opt.num_stage, node_n=48
# linear_size=256 dct_n=35 opt.dropout=0.5 opt.num_stage=12

class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks有多少个剩余块
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        # 输入层到隐含层
        self.fc1 = nn.Linear(72 * input_feature, node_n * input_feature)
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        # 有几个剩余块，就是几个隐含层，hidden_feature 输入，hidden_feature输出
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        # 隐含层到输出层hidden_feature作为输入，input_feature作为输出
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        b, _, f = x.shape
        x = self.fc1(x.reshape(b, -1)).view(b, -1, f)
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, -1, f)
        y = self.act_f(y)
        y = self.do(y)
        for i in range(self.num_stage):
            y = self.gcbs[i](y)
        y = self.gc7(y)
        y = y + x

        return y


class GCN_pose(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks有多少个剩余块
        :param node_n: number of nodes in graph
        """
        super(GCN_pose, self).__init__()
        self.num_stage = num_stage
        # 输入层到隐含层
        self.fc1 = nn.Linear(72 * input_feature, node_n * input_feature)
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        # 有几个剩余块，就是几个隐含层，hidden_feature 输入，hidden_feature输出
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        # 隐含层到输出层hidden_feature作为输入，input_feature作为输出
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        b, _, f = x.shape
        x = self.fc1(x.reshape(b, -1)).view(b, -1, f)
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, -1, f)
        y = self.act_f(y)
        y = self.do(y)
        for i in range(self.num_stage):
            y = self.gcbs[i](y)
        y = self.gc7(y)
        y = y + x

        return y
