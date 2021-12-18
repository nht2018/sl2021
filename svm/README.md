### requirements

python3

numpy

matplotlib

gurobi（https://www.gurobi.com/）

sklearn

### codes instructions

#### sklearn_svm.py

codes for problem 6.2

using sklearn.svm.SVC as classifier

#### svm_with_regularization.py

codes for problem 6.10

my implementation of SVM, using gurobi solver

some differences to the standard SVM

* regularization  can be to the dual problem


$$
\text{maximize   }g_\lambda(\alpha) = (1-\lambda)\sum_{i=1}^{N} \alpha_i - \frac{1}{2}\sum_{i,j=1}^{N} \alpha_i\alpha_jy_iy_jK(x_i,x_j)\\
\text{subject to  } \sum_{i=1}^{N} \alpha_iy_i=0, 0\leq \alpha_i \leq C, i=1,...,N, 
$$

* penalty in the prime problem can be modified
  $$
  \text{minimize  }L(w;C, p) = \frac{1}{2}||w||_2^2+\frac{C}{p}\sum_{i=1}^N \xi_i^p\\
  \text{subject to  } y_i(w^Tx_i+b)\geq1-\xi_i,i=1,...,N
  $$
  here p can be 1(standard SVM) or 2