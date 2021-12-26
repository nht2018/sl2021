import numpy as np
import pandas as pd
import os
import csv
import random
import queue
import json

PATH = 'E:\\sl2021Project\\protein data'
label0, label1 = 'trans', 'hydro'
labeldic = {'trans': 0, 'hydro': 1}


''' 将数据集随机分为train,validate,test,各占约1/3,数据格式,例如train.json可能为

{'3':{'x':[1,26,12,……],'y':'hydro','len':150,'Id':'103D'},'4':……,'10':……,……}
在当前文件夹中存为json格式,无返回值'''


def generator(path='E:\\sl2021Project\\protein data', filename='\\all.csv'):
    csv_file = open(path+filename,
                    mode='r', encoding='utf-8')

    reader = csv.reader(csv_file)
    traindic, validatedic, testdic = {}, {}, {}
    # 将字符串转换成1~26数字列表
    num = 0
    for item in reader:
        if(item):
            num = num+1
            a = random.randint(0, 2)
            Id, strr, type = item[0], item[1], item[2]
            lst = []
            for s in strr:
                lst.append(ord(s)-64)
            leng = len(lst)
            #lst = np.array(lst)
            item = {'x': lst, 'y': labeldic[type], 'len': leng, 'Id': Id}
            s = str(num)
            if(a == 0):
                traindic[s] = item
            if(a == 1):
                validatedic[s] = item
            if(a == 2):
                testdic[s] = item

    csv_file.close()
    json_str = json.dumps(traindic)
    f = open(PATH+'\\train.json', 'w', encoding='utf-8')
    f.write(json_str)
    f.close()
    json_str = json.dumps(testdic)
    f = open(PATH+'\\test.json', 'w', encoding='utf-8')
    f.write(json_str)
    f.close()
    json_str = json.dumps(validatedic)
    f = open(PATH+'\\validate.json', 'w', encoding='utf-8')
    f.write(json_str)
    f.close()


'''shuffle 函数:1功能:给定csv文件A路径，名称和shufflerate作为压缩系数，将文件中数据中的1/shufflerate提取,
在原路径下形成新文件shuffle.json,并输出两类个数，返回dic形式的值'''


def shuffle(path='E:\\sl2021Project\\protein data', filename='\\all.csv', shufflerate=3000):
    csv_file = open(path+filename,
                    mode='r', encoding='utf-8')

    reader = csv.reader(csv_file)
    # 将字符串转换成1~26数字列表
    num = 0
    dic = {}
    type0, type1 = 0, 0
    for item in reader:
        if(item):
            num = num+1
            a = random.randint(0, shufflerate-1)
            if(a == 1):
                Id, strr, type = item[0], item[1], item[2]
                lst = []
                for s in strr:
                    lst.append(ord(s)-64)
                leng = len(lst)
                #lst = np.array(lst)
                item = {'x': lst, 'y': labeldic[type], 'len': leng, 'Id': Id}
                s = str(num)
                if(type == 'hydro'):
                    if(random.randint(0, 3) != 1):
                        dic[s] = item
                        type1 = type1+1
                else:
                    dic[s] = item
                    type0 = type0+1

    csv_file.close()
    json_str = json.dumps(dic)
    f = open('E:\\python'+'\\shuffle.json', 'w', encoding='utf-8')
    f.write(json_str)
    f.close()
    print('shuffle: ', type0, 'type0 and ', type1, 'type1')
    return json_str


''' 对蛋白质数据集预处理,生成all.csv,仅保留hydro类和trans类数据,并删除除了ID,链条,类型以外所有数据,无返回值'''


def preprocess():
    csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_no_dups.csv',
                    mode='r', encoding='utf-8')
    reader = csv.reader(csv_file)
    queue_hydro, queue_trans = queue.Queue(), queue.Queue()
    # 找出哪些ID对应着Hydro和Trans，用queue存放
    for item in reader:
        if(item[1] == 'HYDROLASE'):
            queue_hydro.put(item[0])
        elif(item[1] == 'TRANSFERASE'):
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


if __name__ == '__main__':
    # preprocess()
    # generator()
    shuffledata = shuffle()
