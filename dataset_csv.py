import numpy as np
import pandas as pd
import os
import csv
import random

PATH = 'E:\\sl2021Project\\twitter sentiment'
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
        data = np.array(data)
        if(form == 'array'):
            return data
        data1 = [l.tolist() for l in data]
        return data1


# 将数据集分为train,validate,test,各占约1/10,约有15~16w行数据,在当前文件夹中存为csv格式
def generator(path=PATH, filename='\\all.csv'):
    csv_file = open(path+filename,
                    mode='r', encoding='utf-8')

    reader = csv.reader(csv_file)
    trainset, validateset, testset = [], [], []
    for item in reader:
        a = random.randint(0, 9)
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

# generator()
#a = Dataset('train')
#b = a.Data('list')
