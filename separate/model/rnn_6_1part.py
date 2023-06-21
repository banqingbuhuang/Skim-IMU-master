from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

import numpy
import torch.nn as nn
import torch
from torch import tanh as act
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
        output, (hidn_1, celln_1) = self.bi_lstm(act(self.fc1(x)), (hid0, cell0))
        hidn_1 = hidn_1.to(self.device)
        celln_1 = celln_1.to(self.device)
        output = F.dropout(act(output), self.dropout, training=self.training)
        output = act(self.fc2(output))
        return output


class RNN_pos(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 batch_size,
                 num_layers,
                 device,
                 dropout):
        super(RNN_pos, self).__init__()

        self.dropout = dropout
        # 网络结构

        self.bilstm = BiLSTM(input_size=input_size,
                             output_size=hidden_size,
                             hidden_size=hidden_size,
                             batch_size=batch_size,
                             num_layers=num_layers,
                             device=device,
                             dropout=dropout)
        self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, ori, acc):
        acc = torch.transpose(acc, 0, 1)
        ori = torch.transpose(ori, 0, 1)
        frame, batch, node, _ = acc.shape
        encoder_input = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2)
        output = self.bilstm(encoder_input)
        output = act(self.fc1(output))

        return output.transpose(0, 1)
