import numpy as np

import gurobipy


class SVM(object):
    def __init__(self, C=1):
        self.C = C

        self.kernel = None
        self.train_data = None
        self.train_label = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha = None
        self.beta = None

    def fit(self, X: list, Y: list, kernel=None, reg=0.0):
        '''
        train the svm
        :param X: input data
        :param Y: labels in {-1, 1}
        :param kernel: Kernel
        :param reg: coefficient of L1 norm on alpha
        :return: alpha
        '''
        if kernel is None:
            def kernel(x1, x2): return np.sum(x1 * x2)
        self.kernel = kernel

        self.train_data = X
        self.train_label = Y
        N = len(Y)
        alpha = self._solve_dual(X, Y, reg=reg)
        # support_vector_index = np.where(alpha > 1e-6)[0]
        # print(support_vector_index)
        # self.support_vectors_ = self.train_data[support_vector_index]
        # self.n_support_ = np.array([np.sum(self.train_label[support_vector_index] == -1),                        np.sum(self.train_label[support_vector_index] == 1)])

        I = []
        for i in range(N):
            if 1e-6 < alpha[i] < self.C - 1e-6:
                I.append(i)
        if len(I) == 0:
            I = np.array([np.argmin(self.C - alpha)])
        b = np.mean([Y[i] - np.sum([alpha[j] * Y[j] * self.kernel(X[i], X[j])
                    for j in range(N)]) for i in I])

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
        correct = 0
        for i in range(num):
            if pred[i] == test_label[i]:
                correct += 1
        return correct / num

    def _solve_dual(self, X: list, Y: list, reg=0.0):
        '''
        use gurobi solver to solve the dual problem
        max  (1-reg) \sum_{i=1}^{N} alpha_i - 0.5 * \sum_{i,j=1}^{N} alpha_i * alpha_j * y_i * y_j * K(x_i,x_j)
        s.t. \sum_{i=1}^{N} alpha_i y_i=0, 0 <= alpha_i <= C, i=1,...,N,
        :return (alpha, C - alpha)
        '''
        model = gurobipy.Model(name='svm dual')
        model.setParam('OutputFlag', False)
        N = len(Y)

        gram = np.array([[Y[i] * Y[j] * self.kernel(X[i], X[j])
                        for j in range(N)] for i in range(N)])
        print(gram[0][5])

        #gram += 1e-3 * np.eye(N)
        I = np.array([[[i, j] for j in range(N)]
                     for i in range(N)]).reshape((-1, 2))
        alpha = model.addVars(N, lb=0, ub=self.C, name='alpha')
        model.addConstr(gurobipy.quicksum(
            [alpha[i] * Y[i] for i in range(N)]) == 0, name='constr')
        model.setObjective(
            (1 - reg) * alpha.sum() - 1 / 2 *
            gurobipy.quicksum([alpha[i] * alpha[j] * gram[i, j]
                              for i, j in I]),
            sense=gurobipy.GRB.MAXIMIZE)
        # 调试
        model.Params.NonConvex = 2
        model.optimize()
        alphaarray = np.array([var.x for var in alpha.values()])
        loss = (1 - reg) * alphaarray.sum() - 1 / 2 * \
            gurobipy.quicksum([alphaarray[i] * alphaarray[j]
                              * gram[i, j] for i, j in I])
        print(loss)
        return np.array([var.x for var in alpha.values()])
