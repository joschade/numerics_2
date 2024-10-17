import numpy as np
import scipy.sparse as sp

def reduced_poisson_matrix(n):
    offdiag = np.ones((n,))
    diag = 2 * offdiag
    data = np.array([offdiag, diag, offdiag])
    offset = np.array([-1, 0, 1])

    return sp.dia_matrix((data, offset), shape=(n, n))



A = reduced_poisson_matrix(7)
print(f'A = {A.toarray()}')