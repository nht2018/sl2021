import numpy as np
import sklearn.datasets
'''
this file contains small datasets for testing purposes
'''

class Watermelon(object):
    def __init__(self):
        self.positive = np.array(
            [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
             [0.403, 0.237], [0.481, 0.149], [0.437, 0.211]])
        self.negative = np.array([
            [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], [0.639, 0.161],
            [0.657, 0.198], [0.360, 0.370], [0.593, 0.042], [0.719, 0.103]])

        self.data = np.concatenate([self.positive, self.negative], axis=0)
        self.label = np.concatenate([np.ones(shape=(self.positive.shape[0],)),
                                     np.zeros(shape=(self.negative.shape[0],))], axis=0)
        self.data_num = len(self.label)
        self.feature_name = ['密度', '含糖量']
        self.feature_num = 2

    def shuffle(self):
        perm = np.random.permutation(self.data_num)
        self.data = self.data[perm]
        self.label = self.label[perm]


class Wine(object):
    def __init__(self):
        dataset = sklearn.datasets.load_wine()
        self.positive = dataset.data[np.where(dataset.target == 1)[0]]
        self.negative = dataset.data[np.where(dataset.target == 0)[0]]
        self.data = dataset.data[np.where(dataset.target < 2)[0]]
        self.label = dataset.target[np.where(dataset.target < 2)[0]]
        self.data_num = len(self.label)

class BreastCancer(object):
    def __init__(self):
        dataset = sklearn.datasets.load_breast_cancer()
        self.positive = dataset.data[np.where(dataset.target == 1)[0]]
        self.negative = dataset.data[np.where(dataset.target == 0)[0]]
        self.data = dataset.data[np.where(dataset.target < 2)[0]]
        self.label = dataset.target[np.where(dataset.target < 2)[0]]
        self.data_num = len(self.label)



if __name__ == '__main__':
    dataset = BreastCancer()
    print(dataset.data.shape, dataset.label.shape)
