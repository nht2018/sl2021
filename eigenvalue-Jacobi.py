import math
matrix = [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 3, -1], [0, 0, -1, 4]]

# 矩阵转置函数


def T(n, A):
    M = [[0 for i in range(n)]for j in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = A[j][i]
    return M

# 方阵乘法函数


def MatMulti(n, A, B):
    C = [[0 for i in range(n)]for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k]*B[k][j]
    return C

# Givens变换矩阵，i1，i2是变换的位置，i1<i2，n是矩阵的阶数。
# ，A是矩阵的二维数组


def Givens(i1, i2, n, A):
    C = [[0 for i in range(n)]for j in range(n)]
    for i in range(n):
        if i != i1 and i != i2:
            C[i][i] = 1
    eta = (A[i2][i2] - A[i1][i1])/(2*A[i1][i2])
    t = 0
    if eta >= 0:
        t = 1/(eta + math.sqrt(1 + eta**2))
    else:
        t = -1/(-eta + math.sqrt(1 + eta**2))
    C[i1][i1] = 1/math.sqrt(1 + t**2)
    C[i2][i2] = C[i1][i1]
    C[i1][i2] = C[i1][i1]*t
    C[i2][i1] = -C[i1][i2]

    return C

# 选择迭代的非对角元，要求其绝对值最大。


def Selection(n, A):
    maxnum = 0
    p1 = 0
    p2 = 0
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[i][j]) > maxnum:
                maxnum = abs(A[i][j])
                p1 = i
                p2 = j
    return min(p1, p2), max(p1, p2)

# Jacobi迭代，n是矩阵的阶数，k是迭代的次数，A是矩阵的二维数组。


def Jacobi(n, k, A):
    B = A
    for i in range(k):
        p, q = Selection(n, B)
        G = Givens(p, q, n, B)
        GT = T(n, G)
        B = MatMulti(n, B, G)
        B = MatMulti(n, GT, B)
        for i in range(n):
            for j in range(n):
                if abs(B[i][j]) < 10**(-15):
                    B[i][j] = 0
    return B


M = Jacobi(4, 10, matrix)
for i in range(4):
    print(M[i])
