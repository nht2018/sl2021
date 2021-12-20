import numpy as np
from .dataset import BreastCancer
from .kernel import Kernel, kernel_linear, kernel_rbf
import gurobipy
import matplotlib.pyplot as plt
from .utils import dataset_split



class SVM(object):
    def __init__(self, C=1e3):
        self.C = C

        self.kernel=None
        self.train_data = None
        self.train_label = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha = None
        self.beta = None


    def fit(self, X:list, Y:list, kernel:Kernel = None, reg=0.0):
        '''
        train the svm
        :param X: input data
        :param Y: labels in {-1, 1}
        :param kernel: Kernel
        :param reg: coefficient of L1 norm on alpha
        :return: alpha
        '''
        if kernel is None:
            kernel = kernel_linear
        self.kernel = kernel
        
        self.train_data = X
        self.train_label = Y
        N = len(Y)
        alpha = self._solve_dual(X, Y, reg=reg)
        #support_vector_index = np.where(alpha > 1e-6)[0]
        #print(support_vector_index)
        #self.support_vectors_ = self.train_data[support_vector_index]
        #self.n_support_ = np.array([np.sum(self.train_label[support_vector_index] == -1),                        np.sum(self.train_label[support_vector_index] == 1)])

        I = []
        for i in range(N):
            if  1e-6< alpha[i] <self.C - 1e-6:
                I.append(i)
        if len(I) == 0:
            I = np.array([np.argmin(self.C - alpha)])
        b = np.mean([Y[i] - np.sum([alpha[j] * Y[j] * self.kernel(X[i], X[j]) for j in range(N)]) for i in I])

        self.intercept_ = b
        self.alpha = alpha
        return alpha


    def predict(self, instance):
        '''
        predict the label of one instance
        :param instance: input
        :return: label in {-1, 1}
        '''
        return np.sign(self.intercept_ + np.sum(
            [self.alpha[i] * self.train_label[i] * self.kernel(instance, self.train_data[i]) for i in
             range(len(self.train_label))]))

    def score(self, test_data, test_label):
        num = len(test_label)
        pred = []
        for i in range(num):
            pred.append(self.predict(test_data[i]))
        correct  = 0
        for i in range(num):
            if pred[i] == test_label[i]:
                correct += 1
        return correct / num

    def _solve_dual(self, X:list, Y:list, reg=0.0):
        '''
        use gurobi solver to solve the dual problem

        max  (1-reg) \sum_{i=1}^{N} alpha_i - 0.5 * \sum_{i,j=1}^{N} alpha_i * alpha_j * y_i * y_j * K(x_i,x_j)

        s.t. \sum_{i=1}^{N} alpha_i y_i=0, 0 <= alpha_i <= C, i=1,...,N,

        :return (alpha, C - alpha)
        '''
        model = gurobipy.Model(name='svm dual')
        model.setParam('OutputFlag', False)
        N = len(Y)
        gram = np.array([[Y[i] * Y[j] * self.kernel(X[i], X[j]) for j in range(N)] for i in range(N)])
        I = np.array([[[i, j] for j in range(N)] for i in range(N)]).reshape((-1, 2))
        alpha = model.addVars(N, lb=0, ub=self.C, name='alpha')
        model.addConstr(gurobipy.quicksum([alpha[i] * Y[i] for i in range(N)]) == 0, name='constr')
        model.setObjective(
            (1 - reg) * alpha.sum() - 1 / 2 * gurobipy.quicksum([alpha[i] * alpha[j] * gram[i, j] for i, j in I]),
            sense=gurobipy.GRB.MAXIMIZE)
        model.optimize()
        return np.array([var.x for var in alpha.values()])


if __name__ == '__main__':
    dataset = BreastCancer()
    np.random.seed(1234)
    train_data, train_label, test_data, test_label, _, _ = dataset_split(dataset.data, dataset.label, test_size=.2,
                                                                   valid_size=0.0)
    svm = SVM(C=1e3,
              #kernel=kernel_rbf(gamma=0.1))
              kernel=kernel_linear()) # linear kernel

    svm.fit(train_data, train_label)
    print('train accuracy=', svm.score(train_data, train_label))
    print('test accuracy=', svm.score(test_data, test_label))
    print(svm.n_support_)


    '''
    svm.fit(train_data, train_label, reg=0.8)
    print('train accuracy=', svm.score(train_data, train_label))
    print('test accuracy=', svm.score(test_data, test_label))
    print(svm.n_support_)
    '''