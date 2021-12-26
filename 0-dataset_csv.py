import numpy as np
import pandas as pd
import os
import csv
import random
import queue
import sklearn.datasets


# trans对应标签为0,negative；hydro对应标签为1,positive
# 写需要存放的文件夹名


class DataSet():

    def __init__(self, dataset_name, path, label0, label1):
        # 写文件夹名
        self.ds_path = path
        dataset_meta = {
            'train': (os.path.join(self.ds_path, 'train.csv'), 73257),
            'validate': (os.path.join(self.ds_path, 'validate.csv'), 73257),
            'test': (os.path.join(self.ds_path, 'test.csv'), 26032)
        }
        files, instances = dataset_meta[dataset_name]
        data = pd.read_csv(files, header=None)
        data = np.array(data)  # 将读取的数据转化为nparray
        self.data_num = len(data)
        lst0, lst1, lst = [], [], []
        # 以下需要引入seq2seq将arrayy[1]氨基酸序列转化成向量
        for arrayy in data:
            l = arrayy[1]
            if arrayy[-1] == label0:
                lst0.append(l)
            elif arrayy[-1] == label1:
                lst1.append(l)
            lst.append(l)
        self.positive = np.array(lst1)
        self.negative = np.array(lst0)
        self.data = np.concatenate([self.positive, self.negative], axis=0)
        self.label = np.concatenate([np.ones(shape=(self.positive.shape[0],)),
                                     np.zeros(shape=(self.negative.shape[0],))], axis=0)

    def getData(self):  # nparray形式,n*1
        return self.data

    def getLabel(self):  # n*1
        return self.label

    def getPos(self):  # n1*1
        return self.positive

    def getNeg(self):  # n0*1
        return self.negative


if __name__ == '__main__':
    a = DataSet('train', path='E:\\sl2021Project\\protein data')
    b = a.getPos()
    c = a.getNeg()
    print(1)
