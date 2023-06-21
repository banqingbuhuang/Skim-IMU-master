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
        # self.bil1 = torch.nn.LSTM(input_size, input_size,
        #                           num_layers=2
        #                           )
        self.rnn_pos = RNN_pos(
            input_frame=input_frame_len,
            input_size=input_size,  # 需要修改
            output_size=mid_size,
            device=device,
            hidden_size=hidden_size_pos,
            num_layers=2,
            dropout=dropout
        )
        self.gcn1 = GC_Block(in_features=output_frame_len, device=device,
                             hid_feature=output_frame_len, adj=adjs[0],
                             p_dropout=dropout, bias=True, node_n=mid_size)
        self.progressive = Progressive(frame_num=output_frame_len,
                                       input_size=mid_size,
                                       device=device,
                                       adjs=adjs,
                                       dropout=dropout
                                       )
        self.all_position = RNN_pos_all(
            input_frame=input_frame_len,
            input_size=mid_2_size + input_size,  # 需要修改
            output_size=output_size,
            hidden_size=128,
            num_layers=2,
            dropout=dropout)
        self.gcn2 = GC_Block(in_features=output_frame_len,
                             hid_feature=output_frame_len, adj=adjs[-1], device=device,
                             p_dropout=dropout, bias=True, node_n=output_size)

    def forward(self, ori, acc):
        encoder_input = torch.cat([ori.flatten(2), acc.flatten(2)], dim=2).transpose(0, 1)
        f, b, _ = encoder_input.shape
        posi_6 = self.rnn_pos(encoder_input)
        y1 = self.gcn1(posi_6.transpose(0, 1)).transpose(0, 1)  # 18
        y = self.progressive(y1)
        posi_24 = self.all_position(y, encoder_input)
        posi_24 = self.gcn2(posi_24.transpose(0, 1)).view(b, f, 24, -1)

        return posi_24


class Progressive(nn.Module):
    def __init__(self, frame_num, device,
                 input_size, adjs, dropout):
        super(Progressive, self).__init__()
        mid_size = [7, 8, 9, 10]
        self.up_7 = Up(frame_num=frame_num,
                       input_size=input_size,
                       output_size=mid_size[0] * 3,
                       adj=adjs[2],
                       device=device,
                       dropout=dropout)
        self.up_8 = Up(frame_num=frame_num,
                       input_size=mid_size[0] * 3,
                       output_size=mid_size[1] * 3,
                       adj=adjs[4],
                       device=device,
                       dropout=dropout)
        self.up_9 = Up(frame_num=frame_num,
                       input_size=mid_size[1] * 3,
                       output_size=mid_size[2] * 3,
                       adj=adjs[6],
                       device=device,
                       dropout=dropout)
        self.up_10 = Up(frame_num=frame_num,
                        input_size=mid_size[2] * 3,
                        output_size=mid_size[3] * 3,
                        adj=adjs[8],
                        device=device,
                        dropout=dropout)

    def forward(self, x):
        y = self.up_7(x)
        y = self.up_8(y)
        y = self.up_9(y)
        y = self.up_10(y)
        return y


class Up(nn.Module):
    def __init__(self, frame_num, device,
                 input_size, output_size, adj, dropout):
        super(Up, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size * frame_num)
        self.up = nn.Linear(in_features=input_size, out_features=3)
        self.gL = GraphConvolution(frame_num, out_features=frame_num, adj=adj, node_n=output_size, device=device,
                                   bias=True)
        self.bn2 = nn.BatchNorm1d(output_size * frame_num)
        self.do = nn.Dropout(dropout)  # dropout层

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

    def forward(self, x):
        f, b, _ = x.shape
        y = self.bn1(x.transpose(0, 1).contiguous().view(b, -1)).view(b, f, -1).transpose(0, 1).contiguous()
        up = self.up(y)
        output = torch.cat([y, up], dim=2).contiguous()
        y = self.gL(output.transpose(0, 1).contiguous())
        y = self.bn2(y.view(b, -1)).view(b, f, -1).transpose(0, 1).contiguous()
        y = relu(y)
        y = self.do(y)
        return y + output


class GraphConvolution(nn.Module):

    # 初始化
    # 输入参数，输出参数，权重，偏移 node_n是节点数量
    def __init__(self, in_features, out_features, device, adj, bias=True, node_n=48, dim=3):
        super(GraphConvolution, self).__init__()  # 不加super会报错
        self.in_features = in_features
        self.out_features=out_features
        self.adj = torch.FloatTensor(adj).to(device)
        self.node = node_n // dim
        '''
        将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的)，
        '''
        self.weight = Parameter(torch.FloatTensor(in_features, in_features))
        self.Ap = Parameter(torch.FloatTensor(self.node, self.node))
        self.M = Parameter(torch.FloatTensor(self.node, self.node))
        self.att = Parameter(torch.FloatTensor(self.node, self.node))
        self.Q = Parameter(torch.FloatTensor(self.node, self.node))
        if bias:
            self.bias = Parameter(torch.FloatTensor(node_n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # Adj + np.random.uniform(0, 0.24, size=Adj.shape)
    # 根据输入的Ap和可学习的Q来得到完整的邻接矩阵
    # self.Ap = self.Ap.cuda()
    # 每个图卷积层都有一个不同的可学习的加权邻接矩阵A，可以使网络适应不同操作的连通性

    # 参数初始化
    # 初始化权重，均匀分布随机生成在-stdv到stdv之间的数字
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.Q.data.uniform_(0.01, 0.24)
        self.Ap.data = self.adj
        self.M.data = self.adj
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向函数
    def forward(self, x):
        b, f, n = x.shape
        self.att.data = self.Ap * self.M + self.Q
        x = x.view(b, f, self.node, -1).transpose(2, 3)
        output = torch.matmul(x, self.att).transpose(2, 3).flatten(2)
        output = torch.matmul(self.weight, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 定义格式
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, hid_feature, device, p_dropout, adj, bias, node_n):
        """
        Define a residual block of GCN定义GCN的剩余块
        """
        super(GC_Block, self).__init__()  # 没有super函数会报错
        self.in_features = in_features
        self.out_features = in_features
        # 下面两个有什么区别呢？
        self.gc1 = GraphConvolution(in_features, out_features=in_features, device=device,
                                    adj=adj, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.gc2 = GraphConvolution(in_features, out_features=in_features, device=device,
                                    adj=adj, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)
        self.do = nn.Dropout(p_dropout)  # dropout层
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

    def forward(self, x):
        b, f, n = x.shape
        y = self.gc1(x)
        y = self.bn1(y.view(b, -1)).view(b, f, n)
        y = relu(y)
        y = self.do(y)
        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, -1, n)
        y = relu(y)
        y = self.do(y)
        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


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
        # self.cell = nn.GRUCell(input_size, hidden_size)
        # self.fc1 = nn.Linear(hidden_size, input_size)
        self.bilstm = BiLSTM(input_size=input_size,
                             output_size=output_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)

    def forward(self, x):
        # decoder = x
        # batchsize = decoder.shape[1]
        # state = torch.zeros(batchsize, self.hidden_size)
        # state = state.to(self.device)
        #
        # outputs = []
        # prev = None
        #
        # def loop_function(prev, i):
        #     return prev
        #     # 此处是遍历T-t次得到这些输出，output是一个一个加上来的。
        #
        # for i, inp in enumerate(decoder):
        #     # 一开始prev是None，所以一开始inp是次数的decoder_inputs[i],然后是output
        #     if loop_function is not None and prev is not None:
        #         inp = loop_function(prev, i)
        #
        #     inp = inp.detach()  # 没有梯度
        #     state = self.cell(inp, state)
        #     # 使用inp+是因为residual残差连接
        #     output = inp + self.fc1(F.dropout(state, self.dropout, training=self.training))
        #
        #     outputs.append(output.view([1, batchsize, self.input_size]))
        #     if loop_function is not None:
        #         prev = output
        #
        # outputs = torch.cat(outputs, 0)
        outputs = self.bilstm(x)
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

        self.bilstm = BiLSTM(input_size=input_size,
                             output_size=output_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)

    def forward(self, x, en):
        encoder_input = torch.cat([x, en], dim=2)
        outputs = self.bilstm(encoder_input)
        return outputs


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
        output, hid = self.bi_lstm(relu(self.fc1(self.dropout(x))), hid)
        output = self.fc2(output)
        return output
