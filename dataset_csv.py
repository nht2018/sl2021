import numpy as np
import pandas as pd
import os
import csv
import random
import queue

PATH = 'E:\\sl2021Project\\protein data'
# 写需要存放的文件夹名


class Dataset():

    def __init__(self, dataset_name):
        # 写文件夹名
        self.ds_path = PATH
        dataset_meta = {
            'train': (os.path.join(self.ds_path, 'train.csv'), 73257),
            'validate': (os.path.join(self.ds_path, 'validate.csv'), 73257),
            'test': (os.path.join(self.ds_path, 'test.csv'), 26032)
        }
        self.file, self.instances = dataset_meta[dataset_name]

    def Data(self, form='array'):  # 数据转化成nparray 或list
        data = pd.read_csv(self.file)
        if(form == 'csvform'):
            return data
            # 原格式
        data = np.array(data)
        if(form == 'array'):
            return data
            # nparray格式
        data1 = [l.tolist() for l in data]
        return data1
        # list格式


# 将数据集分为train,validate,test,各占约1/3,在当前文件夹中存为csv格式,无返回值
def generator(path=PATH, filename='\\all.csv'):
    csv_file = open(path+filename,
                    mode='r', encoding='utf-8')

    reader = csv.reader(csv_file)
    trainset, validateset, testset = [], [], []
    for item in reader:
        a = random.randint(0, 2)
        if(a == 0):
            trainset.append(item)
        if(a == 1):
            validateset.append(item)
        if(a == 2):
            testset.append(item)

    csv_file.close()
    f = open(PATH+'\\train.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    for i in trainset:
        writer.writerow(i)
    f.close()
    f = open(PATH+'\\validate.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    for i in validateset:
        writer.writerow(i)
    f.close()
    f = open(PATH+'\\test.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    for i in testset:
        writer.writerow(i)
    f.close()

# 对蛋白质数据集预处理，仅保留hydro类和trans类数据,并删除除了ID,链条,类型以外所有数据,无返回值


def preprocess():
    csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_no_dups.csv',
                    mode='r', encoding='utf-8')
    reader = csv.reader(csv_file)
    queue_hydro, queue_trans = queue.Queue(), queue.Queue()
    # 找出哪些ID对应着Hydro和Trans，用queue存放
    for item in reader:
        if('HYDRO' in item[1]):
            queue_hydro.put(item[0])
        elif('TRANS' in item[1]):
            queue_trans.put(item[0])
    csv_file.close()
    # 找到对应行，并写入新文件all.csv,预处理完成
    csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_seq.csv',
                    mode='r', encoding='utf-8')
    reader = csv.reader(csv_file)
    f = open(PATH+'\\all.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    hydro, trans = queue_hydro.get(), queue_trans.get()
    for item in reader:
        if(item[0] == hydro):
            lst = [item[0], item[2], 'hydro']
            writer.writerow(lst)
            if(queue_hydro.qsize()):
                hydro = queue_hydro.get()
            else:
                hydro = 'ZZZZZ'
        elif(item[0] > hydro and queue_hydro.qsize()):
            hydro = queue_hydro.get()
        if(item[0] == trans):
            lst = [item[0], item[2], 'trans']
            writer.writerow(lst)
            if(queue_trans.qsize()):
                trans = queue_trans.get()
            else:
                trans = 'ZZZZZ'
        elif(item[0] > trans and queue_trans.qsize()):
            trans = queue_trans.get()
    f.close()
    csv_file.close()


preprocess()
generator()
#a = Dataset('train')
#b = a.Data('list')
