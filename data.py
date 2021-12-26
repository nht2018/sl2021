import json
import torch
from torch._C import device


X_train, Y_train, X_test, Y_test = [], [], [], []

X_trainSVM, Y_trainSVM, X_testSVM, Y_testSVM = [], [], [], []


# replace the file to be loaded here.
with open('train.json', 'r') as jsonfile:
    data = json.load(jsonfile)

for key in data:
    lst = []
    for item in data[key]['x']:
        lst.append([[item]])
    x = torch.tensor(lst, dtype=torch.float32).cuda()
    X_train.append(x)
    y = torch.tensor([(data[key]['y'] - 0.5) * 2], dtype=torch.int64).cuda()
    Y_train.append(y)

for key in data:
    X_trainSVM.append(data[key]['x'])
    Y_trainSVM.append((data[key]['y'] - 0.5)*2)

with open('shuffle.json', 'r') as jsonfile:
    data = json.load(jsonfile)


for key in data:
    lst = []
    for item in data[key]['x']:
        lst.append([[item]])
    x = torch.tensor(lst, dtype=torch.float32).cuda()
    X_test.append(x)
    y = torch.tensor([(data[key]['y'] - 0.5) * 2], dtype=torch.int64).cuda()
    Y_test.append(y)
    X_testSVM.append(data[key]['x'])
    Y_testSVM.append((data[key]['y'] - 0.5)*2)

'''
n_data = 374
X_train, Y_train = X_train[0:n_data], Y_train[0:n_data]
X_trainSVM, Y_trainSVM = X_trainSVM[0:n_data], Y_trainSVM[0:n_data]
'''

if __name__ == '__main__':
    print(len(X_trainSVM))
    print(X_trainSVM[0], Y_trainSVM[0])
    print(type(X_trainSVM))
    print(sum([i == 1 for i in Y_trainSVM]))
    print(sum([i == -1 for i in Y_trainSVM]))
