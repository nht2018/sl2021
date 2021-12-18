import numpy as np


class Kernel(object):
    def __init__(self, kernel_func=None, **kwargs):
        self.params = kwargs
        if kernel_func is None:
            kernel_func = lambda x, y, params: np.dot(x, y)
        self._kernel_func = kernel_func

    def __call__(self, vec_x, vec_y):
        return self._kernel_func(vec_x, vec_y, self.params)

def kernel_linear():
    return Kernel()

def kernel_rbf(gamma=1):
    return Kernel(kernel_func=lambda x, y, params: np.exp(-params["gamma"] * np.linalg.norm(x - y) ** 2),
                    gamma=gamma)



if __name__ == '__main__':
    K = kernel_rbf
    n = 10
    x, y = np.random.randn(n), np.random.randn(n)
    print(K(x, y))
    print(K.params)
