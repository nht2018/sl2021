import numpy as np
import torch
from torch import nn, optim

import time
import os
import random

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from model import Model
from seq2seq import Encoder, Decoder, Seq2seq
from kernel import KernelFunc, KernelSeq2Seq, RBF, KernelLinear
from svm0 import SVM
from data import X_train, Y_train, X_test, Y_test, X_trainSVM, Y_trainSVM, X_testSVM, Y_testSVM
from plot import Plot
# block 7, initialize the model


INPUT_DIM = 1
OUTPUT_DIM = 1
ENC_EMB_DIM = 1
DEC_EMB_DIM = 1
HID_DIM = 16   # this can be adjusted
N_LAYERS = 1   # this can be adjusted
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

TARG_LENGTH = 4  # this can be adjusted

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devivce=", dev)


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
seq = Seq2seq(enc, dec)
rbf = RBF()
lin = KernelLinear()

svm = SVM(C=100)

model = Model(seq, X_train, Y_train, target_length=TARG_LENGTH,
              output_dim=OUTPUT_DIM, xs_test=X_test, ys_test=Y_test).cuda()
print(model)
print(model.alphas.shape)


class myCustom(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output):
        return output


criterion = myCustom()
optimizer = optim.SGD(model.parameters(), lr=1e-6,
                      weight_decay=1e-3, momentum=0.9)

n_round = 3


print("training begin")
for round in range(n_round):
    if round == 0:
        n_epoch = 1
    else:
        # 一步就出结果
        n_epoch = 3
    for epoch in range(n_epoch):
        # zero the parameter gradients
        epoch += 1
        print(f"epoch = {epoch} ")
        optimizer.zero_grad()

        # forward + backward + optimize kernel-svm objective wrt \theta (kernel parameters)
        output = model()
        loss = criterion(output)
        print(f"loss = {loss[0][0]} ")
        loss.backward()
        optimizer.step()

    # call kernel-svm solver to optimize objective wrt alpha (at the same time, passing the current kernel)
    # returns the updated new_alphas
    gamma = 1e-3
    new_alphas = svm.fit(model.X_np,
                         Y_trainSVM,
                         # kernel=lambda x1,x2:np.sum(x1 * x2))
                         kernel=lambda x1, x2: np.exp(-gamma * np.linalg.norm(x1-x2)**2))

    print(f"svm score is: {svm.score(model.X_np, Y_trainSVM)}\n")
    # -------------------------------------------------------------
    alpha_list = []
    for item in new_alphas:
        alpha_list.append([item])
    #print(f"new alphas are: {torch.tensor(alpha_list)}")
    model.update_alpha(torch.tensor(alpha_list).cuda())

    X = []
    for lst in model.X_np:
        llst = []
        for i in lst:
            llst.append(i[0][0])
        X.append(llst)
    Plot(X=X, Y=Y_trainSVM)


print('test begin')
print(f"svm score is: {svm.score(model.gen_gram_test(), Y_testSVM)}\n")
Plot(X=model.X_np)
