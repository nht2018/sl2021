import numpy as np
import torch
from torch import nn, optim, tensor

import time
import os
import random
from torch._C import device

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from seq2seq import Seq2seq
from kernel import RBF, KernelLinear, KernelSeq2Seq


class Model(nn.Module):
    def __init__(self, seq: Seq2seq, xs, ys, target_length, output_dim,
                 kernel=RBF(), xs_test=None, ys_test=None):
        super().__init__()
        self.seq = seq
        self.kernel = kernel
        self.kernel_seq2seq = KernelSeq2Seq(
            seq, kernel, target_length, output_dim)
        # xs is a list of input data x,
        # where the size of x is: [sequence len, batch size, feature size]
        self.xs = xs
        # ys is a list of label y.
        # size of y: [batch size]
        self.ys = ys
        self.xs_test = xs_test
        self.ys_test = ys_test
        # data_length: num of items
        # target_length: seq2seq2 output sequence length
        self.target_length = target_length
        # output_dim: seq2seq2 output embedding size; in our case being 1
        self.output_dim = output_dim

        self.data_length = len(ys)
        # size of alphas: [num of items, batch size]
        self.alphas = torch.randn(self.data_length, 1).cuda()

        self.gram = None

    def forward(self):
        self.gen_gram()
        print(self.gram[0][5])
        value = torch.zeros(1, 1).cuda()

        for i in range(self.data_length):
            for j in range(self.data_length):
                # the i-j term of dual kernal-svm objective
                term = self.alphas[i] * self.alphas[j] * self.gram[i][j]
                value = torch.add(value, term)

        sumalpha = 0
        for i in range(0, self.data_length):
            sumalpha += self.alphas[i]
        value = 0.5 * value - sumalpha

        #print(f"objective value = {value[0][0]}")

        return value

        # return 0.5 * torch.sum(torch.tensor([[self.alphas[i] * self.alphas[j] * self.ys[i] * self.ys[j] * self.kernel(self.xs[i], self.xs[j])
        #                                 for j in range(self.data_length)] for i in range(self.data_length)]))
    def gen_gram(self):
        X = [self.seq2seq(self.xs[i]) for i in range(self.data_length)]
        self.X_np = [x.cpu().detach().numpy() for x in X]
        '''之前没有y'''
        self.gram = [[self.ys[i]*self.ys[j]*self.kernel(X[i], X[j]) for i in range(
            self.data_length)] for j in range(self.data_length)]

    def seq2seq(self, x):
        return self.seq(x, torch.zeros(self.target_length, 1, self.output_dim).cuda())

    def update_alpha(self, alphas):
        self.alphas = alphas

    def gen_gram_test(self):
        X = [self.seq2seq(self.xs_test[i]) for i in range(len(self.xs_test))]
        X_np = [x.cpu().detach().numpy() for x in X]
        self.Test_np = X_np
        return X_np
