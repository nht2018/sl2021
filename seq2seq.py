import numpy as np
import torch
from torch import nn, optim

import time
import os
import random

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


# Block 1: Encoder
class Encoder(nn.Module):
    def __init__(self,
                 input_size=1,
                 embedding_size=1,
                 hidden_size=16,
                 n_layers=2,
                 dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size, bias=True)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data,
        size of x: [sequence len, batch size, feature size]
        """

        # size of embedded : [sequence len, batch size, embedding size]
        embedded = self.dropout(F.relu(self.linear(x)))

        output, (hidden, cell) = self.rnn(embedded)
        # hidden: the last step hidden of each layer of rnn
        # size of hidden : [num of layers * num directions, batch size, hidden size]
        # num of directions is 1, since we are useing signle directional rnn
        # cell: the last step cell of each layer of rnn
        # size of cell: [num of layers * num of directions, batch size, hidden size]

        return hidden, cell


# Block 2: Decoder
class Decoder(nn.Module):
    def __init__(self,
                 output_size=1,
                 embedding_size=1,
                 hidden_size=16,
                 n_layers=2,
                 dropout=0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size,
                           n_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x: input batch data,
        size of x: [batch size, feature size]
        x is only 2-dimensional, since the input is batches of last coordinate of the sequence,
        so the sequence length has been removed
        """

        # add a sequence dimension to the front of x, to allow for use of nn.LSTM method
        x = x.unsqueeze(0)
        # size(x) now becomes [1, batch size, feature size]
        embedded = self.dropout(F.relu(self.embedding(x)))

        # size of output : [seq len, batch size, hidden dimension * num of directions]
        # size of hidden : [num of layers * num of directions, batch size, hidden dim]
        # size of cell : [num of layers * num of directions, batch size, hidden dim]

        # notice that sequence len and num of directions will always be 1 in the Decoder, therfore:
        # size of output : [1, batch size, hidden dimension]
        # size of hidden : [num of layers, batch size, hidden dim]
        # size of cell : [num of directions, batch size, hidden dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell


# Block 3: Seq2seq
class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """
        size of x : [observed sequence len, batch size, feature size]
        size of y : [target sequence len, batch size, feature size]
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]

        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).cuda()

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)

        # first input to decoder is last coordinates of x
        decoder_input = torch.Tensor([[0]]).cuda()

        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            # teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = output

        return outputs
