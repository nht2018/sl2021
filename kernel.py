import numpy as np
import torch
from torch import nn, optim

import time
import os
import random
from torch._C import device

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


class KernelFunc(object):
    def __init__(self, kernel_func=None, **kwargs):
        self.params = kwargs
        if kernel_func is None:
            def kernel_func(x, y, params): return np.dot(x, y)
        self._kernel_func = kernel_func

    def __call__(self, vec_x, vec_y):
        return self._kernel_func(vec_x, vec_y, self.params)


# class Kernel refers to the structure combining Seq2seq module and RBF module.
class KernelSeq2Seq(nn.Module):
    def __init__(self, seq2seq, kernel, target_length, output_dim):
        super().__init__()
        # target_length: seq2seq2 output sequence length
        self.target_length = target_length
        # output_dim: seq2seq2 output embedding size; in our case being 1
        self.output_dim = output_dim

        self.seq2seq = seq2seq
        self.kernel = kernel
        self.output_size = torch.randn(self.target_length, 1, self.output_dim)

    def __call__(self, x1, x2):
        """
        size of x1/x2 : [observed sequence len, batch size, feature size]
        """
        # size of output_size corresponds to the size of seq2seq output

        outputs1 = self.seq2seq(x1, self.output_size)
        outputs2 = self.seq2seq(x2, self.output_size)

        # size of value : [batch size, feature size], both being 1 in our case
        value = self.kernel(outputs1, outputs2)

        return value


# Block 4: RBF
class RBF(nn.Module):
    def __init__(self):
        super(RBF, self).__init__()
        #self.sigma = nn.Parameter(torch.Tensor(1))
        self.sigma = 1e3
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.sigma, 1)

    def forward(self, x1, x2):
        '''
        size of x1/x2 : [input sequence len, batch size, feature size],
        for our task, the last two sizes are both 1.
        '''

        value = (x1 - x2).pow(2).sum(0) / self.sigma
        ten = torch.exp(-value)
        ten = ten.cuda()
        return ten


class KernelLinear(nn.Module):
    def __init__(self):
        super(KernelLinear, self).__init__()

    def forward(self, x1, x2):
        return (x1 * x2).sum()
