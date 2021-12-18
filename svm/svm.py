import numpy as np
from dataset import BreastCancer
from kernel import Kernel, kernel_linear, kernel_rbf
import gurobipy
import matplotlib.pyplot as plt
from utils import dataset_split



class SVM(object):
    def __init__(self, C=1e3, kernel:Kernel = None):
        self.C = C
        if kernel is None:
            kernel = kernel_linear
        self.kernel = kernel

        self.train_data = None
        self.train_label = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha = None
        self.beta = None

    @staticmethod
    def label_transform(Y):
        '''
        transform {0, 1} label into {-1, 1}
        '''
        if ((Y-0)*(Y-1) == 0).all() :
            Y = (Y - 0.5) * 2
        return Y.astype(np.int32)

    def fit(self, X, Y, reg=0.0):
        '''
        train the svm
        :param X: input data
        :param Y: labels in {-1, 1}
        :param reg: coefficient of L1 norm on alpha
        :return: None
        '''
        Y = SVM.label_transform(Y)
        self.train_data = X
        self.train_label = Y
        N = len(Y)
        alpha, beta = self._solve_dual(X, Y, reg=reg)
        support_vector_index = np.where(alpha > 1e-6)[0]
        self.support_vectors_ = self.train_data[support_vector_index]
        self.n_support_ = np.array([np.sum(self.train_label[support_vector_index] == -1),
                                    np.sum(self.train_label[support_vector_index] == 1)])

        I = []
        for i in range(len(self.train_label)):
            if beta[i] > 1e-6 and alpha[i] > 1e-6:
                I.append(i)
        if len(I) == 0:
            I = np.array([np.argmin(beta)])
        b = np.mean([Y[i] - np.sum([alpha[j] * Y[j] * self.kernel(X[i], X[j]) for j in range(N)]) for i in I])

        self.intercept_ = b
        self.alpha = alpha
        self.beta = beta


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
        test_label = SVM.label_transform(test_label)
        num = len(test_label)
        pred = []
        for i in range(num):
            pred.append(self.predict(test_data[i]))
        pred = np.array(pred).reshape(test_label.shape)
        correct = np.sum(pred == test_label)
        return correct / num

    def _solve_dual(self, X, Y, reg=0.0):
        '''
        use gurobi solver to solve the dual problem

        max  (1-reg) \sum_{i=1}^{N} alpha_i - 0.5 * \sum_{i,j=1}^{N} alpha_i * alpha_j * y_i * y_j * K(x_i,x_j)

        s.t. \sum_{i=1}^{N} alpha_i y_i=0, 0 <= alpha_i <= C, i=1,...,N,

        :return (alpha, C - alpha)
        '''
        model = gurobipy.Model(name='svm dual')
        model.setParam('OutputFlag', False)
        Y = SVM.label_transform(Y)
        N = Y.shape[0]
        gram = np.array([[Y[i] * Y[j] * self.kernel(X[i], X[j]) for j in range(N)] for i in range(N)])
        I = np.array([[[i, j] for j in range(N)] for i in range(N)]).reshape((-1, 2))
        alpha = model.addVars(N, lb=0, ub=self.C, name='alpha')
        model.addConstr(gurobipy.quicksum([alpha[i] * Y[i] for i in range(N)]) == 0, name='constr')
        model.setObjective(
            (1 - reg) * alpha.sum() - 1 / 2 * gurobipy.quicksum([alpha[i] * alpha[j] * gram[i, j] for i, j in I]),
            sense=gurobipy.GRB.MAXIMIZE)
        model.optimize()
        return np.array([var.x for var in alpha.values()]), self.C - np.array([var.x for var in alpha.values()])


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