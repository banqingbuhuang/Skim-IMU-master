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
    def __init__(self, input_size, output_size, hidden_size, batch_size, num_layers, device, dropout):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.bi_lstm = torch.nn.LSTM(hidden_size, hidden_size,
                                     bidirectional=True,
                                     num_layers=self.num_layers
                                     )
        self.fc2 = torch.nn.Linear(in_features=hidden_size * 2, out_features=output_size)

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

    def forward(self, x):
        hid0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size)
        cell0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size)
        # 开始进行RNN传播
        hid0 = hid0.to(self.device)
        cell0 = cell0.to(self.device)
        output, (hidn_1, celln_1) = self.bi_lstm(tanh(self.fc1(x)), (hid0, cell0))
        hidn_1 = hidn_1.to(self.device)
        celln_1 = celln_1.to(self.device)
        output = F.dropout(output, self.dropout, training=self.training)
        output = tanh(self.fc2(output))
        return output


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
                                    bias=True)
        self.graph_8 = Graph_up_RNN(input_node=node_7,
                                    out_node=node_8,
                                    device=device,
                                    adj_part=adjs[3],
                                    adj_all=adjs[4],
                                    dim=dim,
                                    hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    dropout=dropout,
                                    bias=True)
        self.graph_9 = Graph_up_RNN(input_node=node_8,
                                    out_node=node_9,
                                    device=device,
                                    adj_part=adjs[5],
                                    adj_all=adjs[6],
                                    dim=dim,
                                    hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    dropout=dropout,
                                    bias=True)
        self.graph_10 = Graph_up_RNN(input_node=node_9,
                                     out_node=node_10,
                                     device=device,
                                     adj_part=adjs[7],
                                     adj_all=adjs[8],
                                     dim=dim,
                                     hidden_size=hidden_size,
                                     batch_size=batch_size,
                                     dropout=dropout,
                                     bias=True)

    def forward(self, x):
        y = self.graph_6(x)
        y = self.graph_7(y)
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
                 dropout,
                 bias=True):
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

        self.graph = GraphLinear(out_node, out_size=(out_node * dim), device=device, adj=adj)
        self.fc1 = torch.nn.Linear(out_node * dim, out_node * dim)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_node * dim))  # 偏移为输出参数数量
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
        stdv = 1. / math.sqrt(self.out_node * 3)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        frame, batch, _ = x.shape
        y = self.rnn_bi_1(x)
        xyz = y.reshape(frame, batch, self.out_node, -1).contiguous().transpose(2, 3)
        graphxyz = self.graph(xyz)
        y = tanh(F.dropout(graphxyz, self.dropout, training=self.training))
        output = tanh(self.fc1(y))
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
                         output_size=out_node * dim,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_size=batch_size,
                         device=device,
                         dropout=dropout)
        self.graph_part = GraphLinear(out_node, out_size=(out_node * dim), device=device, adj=adj_part)
        self.even = BiLSTM(input_size=out_node * dim,
                           output_size=out_node * dim,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_size=batch_size,
                           device=device,
                           dropout=dropout)
        self.graph_all = GraphLinear(out_node, out_size=(out_node * dim), device=device, adj=adj_all)
        self.fc1 = torch.nn.Linear(self.out_node * dim, self.out_node * dim)

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
        xyz = y.reshape(frame, batch, self.out_node, -1).transpose(2, 3)
        y = self.graph_part(xyz)
        y = tanh(F.dropout(y, self.dropout, training=self.training))
        y = self.even(y)
        xyz = y.reshape(frame, batch, self.out_node, -1).transpose(2, 3)
        y = self.graph_all(xyz)
        y = tanh(F.dropout(y, self.dropout, training=self.training))
        output = tanh(self.fc1(y))
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
