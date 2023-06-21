from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

import numpy
import torch.nn as nn
import torch
from torch import tanh
from torch.nn.parameter import Parameter
import math
import numpy as np
import torch.nn.functional as F
from utils.opt import Options


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


class RNN_pos(nn.Module):
    def __init__(self, frame_num,
                 input_size,
                 output_size,
                 hidden_size,
                 batch_size,
                 num_layers,
                 device,
                 dropout):
        super(RNN_pos, self).__init__()
        # 网络结构
        self.bilstm = BiLSTM(input_size=input_size,
                             output_size=output_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)

    def forward(self, ori, acc):
        encoder_input = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2)

        y = self.bilstm(encoder_input)
        return y


class Predict_imu(nn.Module):
    def __init__(self,
                 input_frame_len,
                 output_frame_len,
                 input_size,
                 output_size,  # 54
                 batch_size,
                 dim,
                 adjs,
                 num_layers,
                 device,
                 dropout=0.2):
        super(Predict_imu, self).__init__()
        self.input_frame_len = input_frame_len
        self.output_frame_len = output_frame_len

        hidden_size_pos = 256
        hidden_size_progressive = 256
        hidden_size_largeup = 512
        node_10, node_24 = 10, 24
        self.rnn_pos = RNN_pos(frame_num=input_frame_len,
                               input_size=input_size,  # 需要修改
                               output_size=output_size,
                               hidden_size=hidden_size_pos,
                               batch_size=batch_size,
                               num_layers=num_layers,
                               device=device,
                               dropout=dropout)

        self.progressive = GL_RNN_Block(batch_size=batch_size,
                                        hidden_size=hidden_size_progressive,
                                        adjs=adjs,
                                        dim=dim,
                                        device=device,
                                        dropout=dropout)
        self.large_up = Graph_largeup_RNN(input_node=node_10,
                                          out_node=node_24,
                                          device=device,
                                          adj=adjs[-1],
                                          dim=dim,
                                          hidden_size=hidden_size_largeup,
                                          batch_size=batch_size,
                                          dropout=dropout,
                                          bias=True)

    def forward(self, ori, acc):
        acc = torch.transpose(acc, 0, 1)
        ori = torch.transpose(ori, 0, 1)
        posi_6 = self.rnn_pos(ori, acc)

        posi_10 = self.progressive(posi_6)

        frame, batch, _ = posi_10.shape
        rot_24 = self.large_up(posi_10, ori, acc).reshape(frame, batch, 24, -1).contiguous()
        return rot_24.transpose(0, 1)


class GL_RNN_Block(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 adjs,
                 dim,
                 device,
                 dropout,
                 ):
        super(GL_RNN_Block, self).__init__()
        node_6, node_7, node_8, node_9, node_10 = 6, 7, 8, 9, 10

        # 网络结构
        self.graph_6 = Graph_RNN(input_node=node_6,
                                 out_node=node_6,
                                 device=device,
                                 adj=adjs[0],
                                 dim=dim,
                                 hidden_size=hidden_size,
                                 batch_size=batch_size,
                                 dropout=dropout
                                 )
        self.graph_7 = Graph_up_RNN(input_node=node_6,
                                    out_node=node_7,
                                    device=device,
                                    adj_part=adjs[1],
                                    adj_all=adjs[2],
                                    dim=dim,
                                    hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    dropout=dropout,
                                    bias=False)
        self.graph_8 = Graph_up_RNN(input_node=node_7,
                                    out_node=node_8,
                                    device=device,
                                    adj_part=adjs[3],
                                    adj_all=adjs[4],
                                    dim=dim,
                                    hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    dropout=dropout,
                                    bias=False)
        self.graph_9 = Graph_up_RNN(input_node=node_8,
                                    out_node=node_9,
                                    device=device,
                                    adj_part=adjs[5],
                                    adj_all=adjs[6],
                                    dim=dim,
                                    hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    dropout=dropout,
                                    bias=False)
        self.graph_10 = Graph_up_RNN(input_node=node_9,
                                     out_node=node_10,
                                     device=device,
                                     adj_part=adjs[7],
                                     adj_all=adjs[8],
                                     dim=dim,
                                     hidden_size=hidden_size,
                                     batch_size=batch_size,
                                     dropout=dropout,
                                     bias=False)

    def forward(self, x):
        # y = self.graph_6(x)

        y = self.graph_7(x)

        y = self.graph_8(y)
        y = self.graph_9(y)
        y = self.graph_10(y)
        return y


class Graph_RNN(nn.Module):
    def __init__(self,
                 input_node,
                 out_node,
                 adj,
                 hidden_size,
                 device,
                 dim,
                 batch_size,
                 dropout, bias=False):
        super(Graph_RNN, self).__init__()
        self.input_node = input_node
        self.out_node = out_node
        self.dropout = dropout

        self.rnn_bi_1 = BiLSTM(input_size=input_node * dim,
                               output_size=out_node * dim,
                               hidden_size=hidden_size,
                               batch_size=batch_size,
                               num_layers=1,
                               device=device,
                               dropout=dropout)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_node * dim))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def forward(self, x):
        frame, batch, _ = x.shape
        y = self.rnn_bi_1(x)
        xyz = y.reshape(frame, batch, self.out_node, -1).contiguous().transpose(2, 3)
        graphxyz = self.graph(xyz)
        y = tanh(F.dropout(graphxyz, self.dropout, training=self.training))
        output = y
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output


class Graph_up_RNN(nn.Module):
    def __init__(self, input_node,
                 out_node,
                 hidden_size,
                 batch_size,
                 adj_part,
                 dim,
                 adj_all,
                 device,
                 dropout,
                 bias):
        super(Graph_up_RNN, self).__init__()
        self.input_node = input_node
        self.out_node = out_node
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.adj_part = adj_part
        self.adj_all = adj_all

        self.up = BiLSTM(input_size=input_node * dim,
                         output_size=input_node * dim,
                         hidden_size=input_node * dim,
                         num_layers=1,
                         batch_size=batch_size,
                         device=device,
                         dropout=dropout)
        self.even = BiLSTM(input_size=input_node * dim,
                           output_size=out_node * dim,
                           hidden_size=out_node * dim,
                           num_layers=1,
                           batch_size=batch_size,
                           device=device,
                           dropout=dropout)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_node * dim))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_node)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        frame, batch, _ = x.shape
        y = self.up(x)

        y = self.even(y)

        output = y
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output


class GraphLinear(nn.Module):
    def __init__(self, input_node, out_size, device, adj, bias=True):
        super(GraphLinear, self).__init__()
        self.input_node = input_node
        self.out_size = out_size
        self.Adj = Parameter(torch.FloatTensor(adj)).to(device)
        self.M = torch.FloatTensor(adj).to(device)
        self.Q = Parameter(torch.FloatTensor(input_node, input_node)).to(device)
        self.A = Parameter(torch.FloatTensor(input_node, input_node)).to(device)
        self.att = Parameter(torch.FloatTensor(input_node, input_node)).to(device)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    #     self.init_weight()
    #
    # def init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data)
    #             m.bias.data.fill_(0.1)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight.data)
    #             m.bias.data.fill_(0.1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_node * 9)
        self.Q.data.uniform_(0.01, 0.24)
        self.A.data = (self.Adj.data * self.M.data) + self.Q.data
        self.att.data = self.get_laplacian(graph=self.A.data, normalize=True)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def forward(self, pos):
        out = torch.matmul(pos, self.att)
        out = out.transpose(2, 3).flatten(2)
        if self.bias is not None:
            return tanh(out) + self.bias
        else:
            return tanh(out)


class Graph_largeup_RNN(nn.Module):
    def __init__(self, input_node, out_node, device, adj, dim, hidden_size, batch_size, dropout, bias=True):
        super(Graph_largeup_RNN, self).__init__()
        self.input_node = input_node
        self.out_node = out_node
        self.dropout = dropout
        self.dim = dim
        self.posi_rot_10 = BiLSTM(input_size=input_node * dim + 6 * 12,
                                  output_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=2,
                                  batch_size=batch_size,
                                  device=device,
                                  dropout=dropout)
        self.rot_24 = BiLSTM(input_size=hidden_size,
                             output_size=out_node * dim,
                             hidden_size=hidden_size,
                             num_layers=3,
                             batch_size=batch_size,
                             device=device,
                             dropout=dropout)
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_node * dim))  # 偏移为输出参数数量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_node * self.dim)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, pos_10, ori, acc):
        frame, batch, _ = pos_10.shape
        x = torch.cat([pos_10, ori.flatten(2), acc.flatten(2)], dim=2)
        y = self.posi_rot_10(x)
        y = tanh(F.dropout(y, self.dropout, training=self.training))
        y = self.rot_24(y)
        y = tanh(F.dropout(y, self.dropout, training=self.training))

        output = y
        if self.bias is not None:
            return output + self.bias  # 如果有偏移就加上偏移值
        else:
            return output
