# 先调入需要的模块

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sb
from sklearn.decomposition import PCA


def Plot(X=[[0.1, 0.7],
            [0.3, 0.6],
            [0.4, 0.1],
            [0.5, 0.4],
            [0.8, 0.04],
            [0.42, 0.6],
            [0.9, 0.4],
            [0.6, 0.5],
            [0.7, 0.2],
            [0.7, 0.67],
            [0.27, 0.8],
            [0.5, 0.72]], Y=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]):

    X = np.array(X)
    pca = PCA(n_components=2)
    data = pca.fit_transform(X)

    target = Y

    # 定义计算域、文字说明等

    C = 0.0001  # SVM regularization parameter, since Scikit-learn doesn't allow C=0
    # linear_svc = svm.SVC(kernel='linear', C=C).fit(data, target)

    # create a mesh to plot in

    x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
    y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
    if (len(data) >= 10):
        cut_off = int(len(data)/10)
        dax = np.sort(data[:, 0])
        x_min, x_max = dax[cut_off], dax[-cut_off-1]
        day = np.sort(data[:, 1])
        y_min, y_max = day[cut_off], day[-cut_off-1]

    h = max((x_max-x_min)/1000, 0.000002)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    # RBF Kernel

    plt.figure(figsize=(16, 15))

    for i, gamma in enumerate([0.001]):
        rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(data, target)

        # ravel - flatten
        # c_ - vstack
        # #把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
        Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(3, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)

        # Plot the training points
        for i in range(len(data)):
            if(x_min <= data[i][0] <= x_max and y_min <= data[i][1] <= y_max):
                if(Y[i] == 0):
                    plt.scatter(data[i][0], data[i][1],
                                marker='o', color='r', s=1, lw=3)
                elif(Y[i] == 1):
                    plt.scatter(data[i][0], data[i][1],
                                marker='x', color='k', s=1, lw=3)

        plt.title('RBF SVM with $\gamma=$' + str(gamma))

        plt.show()


if __name__ == '__main__':
    Plot()
