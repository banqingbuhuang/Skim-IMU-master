import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch import nn
import torch.nn.functional as F


class Seq2SeqModel(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self, input_seq,
                 target_seq,
                 rnn_size,
                 output_size,
                 input_size,
                 device,
                 one_hot=True,
                 dropout=0.5):
        super(Seq2SeqModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.device = device
        self.fc1 = nn.Linear(self.input_size, self.output_size)
        self.cell = torch.nn.GRUCell(self.output_size, self.rnn_size)
        self.fc2 = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, encoder_inputs, decoder_inputs):

        batchsize = encoder_inputs.shape[0]
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        encoder_inputs = self.fc1(encoder_inputs)
        state = torch.zeros(batchsize, self.rnn_size)  # 输出状态？
        state = state.to(self.device)
        for i in range(self.input_seq - 1):
            state = self.cell(encoder_inputs[i], state)
            #        state2 = self.cell2(state, state2)
            state = F.dropout(state, self.dropout, training=self.training)
            state = state.to(self.device)
        #            state2 = state2.cuda()

        outputs = []
        prev = None

        def loop_function(prev, i):
            return prev

        inp = encoder_inputs[-1]
        # 此处是遍历T-t次得到这些输出，output是一个一个加上来的。
        for i in range(self.target_seq):
            # 一开始prev是None，所以一开始inp是次数的decoder_inputs[i],然后是output
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()  # 没有梯度

            state = self.cell(inp, state)

            output = inp + self.fc2(F.dropout(state, self.dropout, training=self.training))
            state = state.to(self.device)
            outputs.append(output.view([1, batchsize, self.output_size]))
            if loop_function is not None:
                prev = output

        #    return outputs, state

        outputs = torch.cat(outputs, 0)
        return torch.transpose(outputs, 0, 1)
