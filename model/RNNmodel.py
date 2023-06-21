from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
import torch.nn.functional as F
from utils.opt import Options


class RNN_pos(nn.Module):
    def __init__(self,
                 input_frame_len,
                 input_size,
                 rnn_size,
                 batch_size,
                 num_layers,
                 device,
                 dropout=0.2,
                 ):
        super(RNN_pos, self).__init__()
        """
        Args:
            input_frame_len:输入帧数
            output_frame_len:输出帧数
            input_size:输入维度72
            out_size:输出维度18
            batch_size:
        """
        assert len(num_layers) == 2
        # 训练的参数
        self.input_frame_len = input_frame_len
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.num_layers_1 = num_layers[0]
        self.num_layers_2 = num_layers[1]
        self.device = device
        self.dropout = dropout

        # RNN 循环
        # 双向LSTM
        self.bi_lstm = torch.nn.LSTM(self.input_size,
                                     self.rnn_size,
                                     bidirectional=True,
                                     num_layers=self.num_layers_1
                                     )
        self.dan_lstm = torch.nn.LSTM(self.rnn_size * 2,
                                      self.rnn_size,
                                      num_layers=self.num_layers_2
                                      )
        self.linear = torch.nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, encoder_inputs):
        batchsize = encoder_inputs.shape[0]
        # batch,frame,dim->frame,batch,input_dim
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        hid0_1 = torch.zeros(self.num_layers_1 * 2, batchsize, self.rnn_size)
        cell0_1 = torch.zeros(self.num_layers_1 * 2, batchsize, self.rnn_size)
        # 开始进行RNN传播
        hid0_1 = hid0_1.to(self.device)
        cell0_1 = cell0_1.to(self.device)
        output, (hidn_1, celln_1) = self.bi_lstm(encoder_inputs, (hid0_1, cell0_1))
        hidn_1 = hidn_1.to(self.device)
        celln_1 = celln_1.to(self.device)
        output = F.dropout(output, self.dropout, training=self.training)
        # 第二个单向LSTM的
        hid0_2 = torch.zeros(self.num_layers_2, batchsize, self.rnn_size)
        cell0_2 = torch.zeros(self.num_layers_2, batchsize, self.rnn_size)
        hid0_2 = hid0_2.to(self.device)
        cell0_2 = cell0_2.to(self.device)
        output, (hidN_2, cellN_2) = self.dan_lstm(output, (hid0_2, cell0_2))
        hidN_2 = hidN_2.to(self.device)
        cellN_2 = cellN_2.to(self.device)
        output = F.dropout(output, self.dropout, training=self.training)
        return output


class Predict_imu(nn.Module):
    def __init__(self,
                 input_frame_len,
                 output_frame_len,
                 input_size,
                 rnn_size,
                 batch_size,
                 adjs,
                 dim,
                 num_layers,
                 device,
                 dropout=0.2,
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
        assert len(num_layers) == 2
        # 训练的参数
        self.input_frame_len = input_frame_len
        self.output_frame_len = output_frame_len
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.adjs = adjs
        self.num_layers_1 = num_layers[0]
        self.num_layers_2 = num_layers[1]
        self.device = device
        self.dropout = dropout
        nodes = [6, 7, 8, 9, 10, 24]

        # 网络结构:
        # 1.acc+ori数据转换为position数据，得到6*3=18位的数据
        # 2.18位数据先通过一个graphLinear建立结构信息，然后通过双向的LSTM得到时间信息，再通过graphlinear得到信息的补充。
        #
        # 节点为6
        self.graph_n_6 = GraphLinear(input_node=nodes[0],
                                     rnn_size=rnn_size,
                                     device=device,
                                     adj=adjs[0])
        self.rnn_pos = RNN_pos(input_frame_len=input_frame_len,
                               input_size=input_size,
                               rnn_size=rnn_size,
                               batch_size=batch_size,
                               num_layers=num_layers,
                               device=device,
                               dropout=dropout
                               )
        # 输出frame,batch,dim=18
        self.progressive = GL_RNN_Bloack(nodes, input_frame_len, output_frame_len, batch_size, rnn_size, device,
                                         dim=dim, adjs=adjs,
                                         num_layers=self.num_layers_2, dropout=dropout)

    def forward(self, x):
        y = self.rnn_pos(x)
        y = self.progressive(y)
        return  torch.transpose(y, 0, 1)


class GL_RNN_Bloack(nn.Module):
    def __init__(self, nodes, input_frame_len, output_frame_len, batch_size, rnn_size, device, dim, adjs, num_layers,
                 dropout):
        super(GL_RNN_Bloack, self).__init__()
        node_6 = nodes[0]
        node_7 = nodes[1]
        node_8 = nodes[2]
        node_9 = nodes[3]
        node_10 = nodes[4]
        node_24 = nodes[5]
        self.batch_size = batch_size
        self.device = device
        self.dim = dim  # 数据格式，一般为3
        self.num_layers = num_layers
        # 节点为6
        self.graph_n_6 = GraphLinear(input_node=node_6,
                                     rnn_size=rnn_size,
                                     device=device,
                                     adj=adjs[0])
        # 节点为7
        self.predict_rnn = Predict_RNN(input_frame_len=input_frame_len,
                                       output_frame_len=output_frame_len,
                                       input_size=rnn_size,
                                       rnn_size=rnn_size,
                                       batch_size=batch_size,
                                       num_layers=num_layers,
                                       device=device,
                                       dropout=dropout
                                       )
        self.graph_n_7 = Graph_up_RNN(input_node=node_6,
                                      out_node=node_7,
                                      dim=dim,
                                      adj_part=adjs[1],
                                      adj_all=adjs[2],
                                      rnn_size=rnn_size,
                                      device=device,
                                      batch_size=batch_size,
                                      dropout=dropout
                                      )
        self.graph_n_8 = Graph_up_RNN(input_node=node_7,
                                      out_node=node_8,
                                      dim=dim,
                                      adj_part=adjs[3],
                                      adj_all=adjs[4],
                                      rnn_size=rnn_size,
                                      device=device,
                                      batch_size=batch_size,
                                      dropout=dropout
                                      )
        self.graph_n_9 = Graph_up_RNN(input_node=node_8,
                                      out_node=node_9,
                                      dim=dim,
                                      adj_part=adjs[5],
                                      adj_all=adjs[6],
                                      rnn_size=rnn_size,
                                      device=device,
                                      batch_size=batch_size,
                                      dropout=dropout
                                      )
        self.graph_n_10 = Graph_up_RNN(input_node=node_9,
                                       out_node=node_10,
                                       dim=dim,
                                       adj_part=adjs[7],
                                       adj_all=adjs[8],
                                       rnn_size=rnn_size,
                                       device=device,
                                       batch_size=batch_size,
                                       dropout=dropout
                                       )
        self.graph_n_24 = Graph_Largeup_RNN(input_node=node_10,
                                            out_node=node_24,
                                            dim=dim,
                                            adj_all=adjs[9],
                                            rnn_size=rnn_size,
                                            device=device,
                                            batch_size=batch_size,
                                            dropout=dropout
                                            )

    def forward(self, x):
        y = self.predict_rnn(x)
        if torch.isnan(y).sum() != 0:
            print("graph_n_6 就已经chuxian nan ")
        # 输入为20,32,18
        # 提前预测
        # 输出为50,32,36

        y = self.graph_n_6(y)
        if torch.isnan(y).sum() != 0:
            print("predict_rnn 就已经chuxian nan ")
        y = self.graph_n_7(y)
        if torch.isnan(y).sum() != 0:
            print("graph_n_7 就已经chuxian nan ")
        y = self.graph_n_8(y)
        y = self.graph_n_9(y)
        n_10 = self.graph_n_10(y)
        y = self.graph_n_24(n_10)
        return n_10, y


class Predict_RNN(nn.Module):
    def __init__(self, input_frame_len, output_frame_len, input_size, rnn_size, batch_size, device, dropout,
                 num_layers):
        super(Predict_RNN, self).__init__()
        self.input_size = input_size  # 18
        self.rnn_size = rnn_size  # 18
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.input_frame_len = input_frame_len  # 输入帧数
        self.output_frame_len = output_frame_len  # 输出帧数

        self.rnn = BiLSTM(input_size, rnn_size, batch_size, num_layers, device, dropout)
        self.linear = torch.nn.Linear(self.rnn_size * 2, self.rnn_size)

    def forward(self, x):
        outputs = torch.tensor([]).to(self.device)
        prev = None

        def loop_function(prev, i):
            return prev

        inp = x
        # x 是 20,32,18
        for i in range(self.output_frame_len):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)
            y = self.rnn(inp)
            y = F.dropout(y, self.dropout, training=self.training)
            # 36->18位
            y = self.linear(y)
            # 残差连接
            y = inp + y
            if outputs.size() == torch.Size([0]):
                outputs = torch.cat((inp, y[-1, :, :].reshape(1, self.batch_size, self.rnn_size)), dim=0)
            else:
                outputs = torch.cat((outputs, y[-1, :, :].reshape(1, self.batch_size, self.rnn_size)), dim=0)
            if loop_function is not None:
                prev = outputs
        return outputs


class BiLSTM(nn.Module):
    def __init__(self, input_size, rnn_size, batch_size, num_layers, device, dropout):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout

        self.bi_lstm = torch.nn.LSTM(input_size,
                                     rnn_size,
                                     bidirectional=True,
                                     num_layers=self.num_layers
                                     )

    def forward(self, x):
        hid0 = torch.zeros(self.num_layers * 2, self.batch_size, self.rnn_size)
        cell0 = torch.zeros(self.num_layers * 2, self.batch_size, self.rnn_size)
        # 开始进行RNN传播
        hid0 = hid0.to(self.device)
        cell0 = cell0.to(self.device)
        output, (hidn_1, celln_1) = self.bi_lstm(x, (hid0, cell0))

        hidn_1 = hidn_1.to(self.device)
        celln_1 = celln_1.to(self.device)
        output = F.dropout(output, self.dropout, training=self.training)
        return output


class GraphLinear(nn.Module):
    def __init__(self, input_node, rnn_size, device, adj, bias=False):
        super(GraphLinear, self).__init__()

        self.input_node = input_node
        self.Adj = torch.as_tensor(adj).to(device)
        self.M = torch.as_tensor(self.Adj).to(device)
        self.Q = Parameter(
            torch.FloatTensor(input_node, input_node)).to(device)
        assert self.Adj.data.shape == self.Q.data.shape
        self.att = Parameter(
            torch.FloatTensor(input_node, input_node)).to(device)
        if bias:
            self.bias = Parameter(torch.FloatTensor(rnn_size))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.Q.data.uniform_(0.01, 0.24)
        self.att = (self.Adj.data * self.M + self.Q.data)

    def forward(self, x):
        frame, batch, _ = x.shape

        x = x.reshape(frame, batch, -1, self.input_node).contiguous()
        output = torch.matmul(x, self.att).reshape(frame, batch, -1).contiguous()
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output


# 输入 20,32,36
class Graph_up_RNN(nn.Module):
    def __init__(self, input_node, out_node, rnn_size, batch_size, dim, device, adj_part, adj_all, dropout, bias=True):
        super(Graph_up_RNN, self).__init__()
        self.input_node = input_node
        self.out_node = out_node
        self.rnn_size = rnn_size
        self.dim = dim  # 3
        self.Adj_part = torch.as_tensor(adj_part).to(device)
        self.Q_1 = Parameter(
            torch.FloatTensor(out_node, out_node)).to(device)
        self.att_part = Parameter(
            torch.FloatTensor(out_node, out_node)).to(device)

        self.Adj_all = torch.as_tensor(adj_all).to(device).to(device)

        self.Q_2 = Parameter(
            torch.FloatTensor(out_node, out_node)).to(device)
        self.att_all = Parameter(
            torch.FloatTensor(out_node, out_node)).to(device)
        assert self.Adj_all.shape == self.Q_1.shape

        self.bilstm_0 = BiLSTM(input_size=input_node * dim,
                               rnn_size=input_node * dim,
                               batch_size=batch_size,
                               num_layers=1,
                               device=device,
                               dropout=dropout)
        self.bilstm_1 = BiLSTM(input_size=input_node * dim * 2,
                               rnn_size=dim,
                               batch_size=batch_size,
                               num_layers=1,
                               device=device,
                               dropout=dropout)
        self.bilstm_2 = BiLSTM(input_size=out_node * dim * 2,
                               rnn_size=out_node * dim,
                               batch_size=batch_size,
                               num_layers=1,
                               device=device,
                               dropout=dropout)
        self.linear = torch.nn.Linear(self.out_node * dim * 2, self.out_node * dim)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_node * dim))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # 参数初始化
    # 初始化权重，均匀分布随机生成在-stdv到stdv之间的数字
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_node * self.dim)
        self.Q_1.data.uniform_(0.01, 0.24)
        self.Q_2.data.uniform_(0.01, 0.3)
        self.att_part = (self.Adj_part.data + self.Q_1.data)
        self.att_all = (self.Adj_all.data + self.Q_2.data)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if torch.isnan(x).sum() != 0:
            print("graph_up_linear de shuru 个就rnn已经chuxian nan ")

        frame, batch, _ = x.shape

        input = self.bilstm_0(x)

        # 输入20，32，36，输出20，32，6
        y = self.bilstm_1(input)

        # 拼接成20，32，42
        y = torch.cat((input, y), dim=2)
        if torch.isnan(y).sum() != 0:
            print("graph_up_linear de cat 个就rnn已经chuxian nan ")
        y = y.reshape(frame, batch, self.dim * 2, self.out_node).contiguous()
        y = torch.matmul(y, self.att_part).reshape(frame, batch,
                                                   -1).contiguous()  # 三个矩阵相乘input X与权重W相乘，然后adj矩阵与 他们的积稀疏乘。(AHW)三个举证相乘

        if torch.isnan(y).sum() != 0:
            print("graph_up_linear de matmul 个就rnn已经chuxian nan ")

        y = self.bilstm_2(y)
        if torch.isnan(y).sum() != 0:
            print("graph_up_linear de bilstm_2 个就rnn已经chuxian nan ")
        y = y.reshape(frame, batch, self.dim * 2, self.out_node).contiguous()
        y = torch.matmul(y, self.att_all).reshape(frame, batch, -1).contiguous()
        if torch.isnan(y).sum() != 0:
            print("graph_up_linear de matmul att_all个就rnn已经chuxian nan ")
        # y = self.bilstm_2(y)
        output = self.linear(y)
        if torch.isnan(y).sum() != 0:
            print("graph_up_linear de linear个就rnn已经chuxian nan ")
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output

    # 图卷积


class Graph_Largeup_RNN(nn.Module):
    def __init__(self, input_node, out_node, dim, adj_all, rnn_size, device, batch_size, dropout, bias=True):
        super(Graph_Largeup_RNN, self).__init__()

        self.input_node = input_node
        self.out_node = out_node
        self.dim = dim
        self.out_node = out_node
        self.Adj = torch.as_tensor(adj_all).to(device)
        self.weight = Parameter(torch.FloatTensor(rnn_size, rnn_size))
        self.Q = Parameter(
            torch.FloatTensor(out_node, out_node).to(device))
        assert self.Adj.shape == self.Q.shape
        self.att = Parameter(
            torch.FloatTensor(out_node, out_node)).to(device)
        self.bilstm_0 = BiLSTM(input_size=input_node * dim,
                               rnn_size=input_node * dim,
                               batch_size=batch_size,
                               num_layers=1,
                               device=device,
                               dropout=dropout)
        self.bilstm_1 = BiLSTM(input_size=input_node * dim * 2,
                               rnn_size=out_node * dim,
                               batch_size=batch_size,
                               num_layers=2,
                               device=device,
                               dropout=dropout)
        self.bilstm_2 = BiLSTM(input_size=out_node * dim * 2,
                               rnn_size=out_node * dim,
                               batch_size=batch_size,
                               num_layers=2,
                               device=device,
                               dropout=dropout)
        self.linear = torch.nn.Linear(self.out_node * dim * 2, self.out_node * dim)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_node * dim))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_node * self.dim)
        self.Q.data.uniform_(0.1, 0.24)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.att = (self.Adj.data + self.Q.data)

    def forward(self, x):
        frame, batch, _ = x.shape
        input = self.bilstm_0(x)
        y = self.bilstm_1(input)

        y = y.reshape(frame, batch, self.dim * 2, self.out_node).contiguous()
        y = torch.matmul(y, self.att).reshape(frame, batch,
                                              -1).contiguous()  # 三个矩阵相乘input X与权重W相乘，然后adj矩阵与 他们的积稀疏乘。(AHW)三个举证相乘
        y = self.bilstm_2(y)
        output = self.linear(y)
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output

#
# option = Options().parse()
# model = Predict_imu(input_frame_len=20, output_frame_len=30, input_size=72, rnn_size=18, batch_size=32,
#                      adjs=option.adjs,dim=3,
#                     num_layers=[2, 3], device='cuda')
# model = model.cuda()
# input = torch.randn(32, 20, 72).cuda()
# output = model(input)
