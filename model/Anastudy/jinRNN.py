from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from torch.nn.functional import relu
import torch.nn.functional as F

"""
这个是rnn_gcn_pro的模型
"""


class Predict_imu(nn.Module):
    def __init__(self,
                 input_frame_len,
                 output_frame_len,
                 input_size,  # 120
                 mid_size,
                 device,
                 output_size,  # 30
                 adjs,
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
        self.dropout = dropout
        # 网络结构
        hidden_size_pos = 256
        mid_2_size = 30
        self.rnn_pre = RNN_GRU(
            input_frame=input_frame_len,
            out_frame=output_frame_len,
            input_size=input_size,
            output_size=input_size,
            device=device,
            dropout=dropout
        )
        self.rnn_pos = RNN_pos(
            input_frame=input_frame_len,
            input_size=input_size,  # 需要修改
            output_size=mid_size,
            device=device,
            hidden_size=hidden_size_pos,
            num_layers=1,
            dropout=dropout
        )
        self.all_position = RNN_pos_all(
            input_frame=input_frame_len,
            input_size=mid_size + input_size,  # 需要修改
            output_size=output_size,
            hidden_size=128,
            num_layers=1,
            dropout=dropout)

    def forward(self, ori, acc):
        encoder_input = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2).transpose(0, 1)
        encoder_input = self.rnn_pre(encoder_input)
        f, b, _ = encoder_input.shape
        posi_6 = self.rnn_pos(encoder_input)
        posi_24 = self.all_position(posi_6, encoder_input)

        return posi_24.transpose(0, 1).view(b, f, 24, -1)


class RNN_GRU(nn.Module):
    def __init__(self,
                 input_frame, out_frame, input_size, output_size, device, dropout):
        super(RNN_GRU, self).__init__()
        self.input_frame = input_frame
        self.input_size = input_size
        self.out_frame = out_frame
        self.device = device
        self.output_size = output_size
        self.dropout = dropout
        self.hid_size = 128
        self.cell = nn.LSTMCell(input_size, self.hid_size)
        self.fc1 = nn.Linear(in_features=self.hid_size, out_features=input_size)

    def forward(self, x):
        encoder_in = x
        batchsize = encoder_in.shape[1]
        input_frame = encoder_in.shape[0]
        Hid = torch.zeros(batchsize, self.hid_size)  # 输出状态？
        C = torch.zeros(batchsize, self.hid_size)  # 输出状态？
        Hid = Hid.to(self.device)
        C = C.to(self.device)
        for i in range(input_frame - 1):
            Hid, C = self.cell(encoder_in[i], (Hid, C))
            #        state2 = self.cell2(state, state2)
            Hid = F.dropout(Hid, self.dropout, training=self.training)
            Hid = Hid.to(self.device)
        #            state2 = state2.cuda()

        inp = encoder_in[-1]
        outputs = []
        prev = None

        def loop_function(prev, i):
            return prev
            # 此处是遍历T-t次得到这些输出，output是一个一个加上来的。

        for i in range(self.out_frame - input_frame):
            # 一开始prev是None，所以一开始inp是次数的decoder_inputs[i],然后是output
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()  # 没有梯度
            Hid, C = self.cell(inp, (Hid, C))
            Hid = Hid.to(self.device)
            # 使用inp+是因为residual残差连接
            output = inp + self.fc1(F.dropout(Hid, self.dropout, training=self.training))

            outputs.append(output.view([1, batchsize, self.input_size]))
            if loop_function is not None:
                prev = output

        outputs = torch.cat(outputs, 0)
        outputs = torch.cat([encoder_in, outputs], dim=0)
        return outputs


class RNN_pos(nn.Module):
    def __init__(self,
                 input_frame,
                 input_size,  # 输入格式
                 output_size,  # 输出格式
                 hidden_size,
                 num_layers,
                 device,
                 dropout):
        super(RNN_pos, self).__init__()
        self.input_frame = input_frame
        self.input_size = input_size
        self.device = device
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # 网络结构

        self.bilstm = BiRNN(input_size=input_size,
                            output_size=output_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

    def forward(self, x):
        encoder_in = x
        outputs = self.bilstm(encoder_in)
        return outputs


class RNN_pos_all(nn.Module):
    def __init__(self,
                 input_frame,
                 input_size,  # 输入格式
                 output_size,  # 输出格式
                 hidden_size,
                 num_layers,
                 dropout):
        super(RNN_pos_all, self).__init__()
        self.input_frame = input_frame
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # 网络结构

        self.bilstm = BiRNN(input_size=input_size,
                            output_size=output_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

    def forward(self, x, en):
        encoder_input = torch.cat([x, en], dim=2)
        outputs = self.bilstm(encoder_input)
        return outputs


class BiRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(BiRNN, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.bilstm = torch.nn.LSTM(hidden_size, hidden_size,
                                    bidirectional=True,
                                    num_layers=num_layers
                                    )
        self.fc2 = torch.nn.Linear(in_features=hidden_size * 2, out_features=output_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, hid=None):
        # 开始进行RNN传播
        output, hid = self.bilstm(relu(self.fc1(self.dropout(x))), hid)
        output = self.fc2(output)
        return output
