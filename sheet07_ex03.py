import numpy as np
import scipy.sparse as sp

import utils
import pandas as pd

from scipy.sparse.linalg import gmres, inv

from numpy import exp
from sheet02_ex03 import l_squared_norm

# set sparse matrix corresponding to central difference
def backdiff_matrix(n: int, h: float) -> sp.csc_matrix:
    diag = offdiag = np.ones((n,))
    data = np.array([-offdiag, diag])
    offset = np.array([-1, 0])

    return 1/h * sp.dia_matrix((data, offset), shape=(n, n)).tocsc()

u = lambda x, epsilon: x - (exp(-(1-x)/epsilon) - exp(-1/epsilon))/(1 - exp(-1/epsilon))
hs = np.array([1/2**k for k in range(3,9)])

epsilons = np.array([10**(-2*i) for i in range(4)])


# This code answers Problems 07.03 b) and 07.03 c) concurrently.
if __name__ == "__main__":
    print('Sheet 07 Question 03 Subproblem b)-c)')

    results = []

    for epsilon in epsilons:
        for h in hs:
            x_grid = utils.grid_from_stepsize(0, 1, h)
            f_grid = np.ones(x_grid.size)
            u_grid = u(x_grid, epsilon)

            A = -epsilon * utils.reduced_poisson_matrix(x_grid.size, h) + backdiff_matrix(x_grid.size, h)

            u_hat_grid = sp.linalg.spsolve(A, f_grid)

            results.append([epsilon, h, l_squared_norm(u_grid, u_hat_grid)])

    df_backwards = pd.DataFrame(results, columns=['epsilon',  'h', 'l_squared_error'])
    print('results for backwards difference:')
    print(f'{df_backwards=}')
    df_backwards.to_csv('sheet_07/df_backwards.csv')

    df_backwards_pivot = df_backwards.pivot(columns='epsilon', index='h', values='l_squared_error')
    df_backwards_pivot.to_csv('sheet_07/df_backwards_pivot.csv')
    print(f'{df_backwards_pivot=}')

# This section answers 07.03 d)

def jacobi(A: np.array, b:np.array, x_0 = None, tol=1e-10, maxiter=1e5) -> np.array:
    D = sp.dia_matrix((A.diagonal(0), 0), shape=A.shape).tocsc()
    L = sp.tril(A, k=-1)
    U = sp.triu(A, k=1)

    def update(x_k):
        return inv(D).dot(b - (L+U).dot(x_k))

    iter = 0
    err = tol
    if x_0 == None: x_k = np.zeros(A.shape[1])
    while err >= tol and iter <= maxiter:
        x_k = update(x_k)
        err = np.linalg.norm(b - A.dot(x_k))
        iter = iter + 1

    return x_k, iter-1, err

def gauss_seidel(A: np.array, b:np.array, x_0 = None, tol=1e-10, maxiter=1e5) -> np.array:
    L = sp.tril(A, k=0)
    U = sp.triu(A, k=1)

    def update(x_k):
        return inv(L).dot(b - U.dot(x_k))

    iter = 0
    err = tol
    if x_0 == None: x_k = np.zeros(A.shape[1])
    while err >= tol and iter <= maxiter:
        x_k = update(x_k)
        err = np.linalg.norm(b - A.dot(x_k))
        iter = iter + 1

    return x_k, iter-1, err


if __name__ == "__main__":
    print('Sheet 07 Question 03 Subproblem d)')

    results_gmres = []
    results_jacobi = []
    results_gauss_seidel = []

    for epsilon in epsilons:
        for h in hs:
            x_grid = utils.grid_from_stepsize(0, 1, h)
            f_grid = np.ones(x_grid.size)
            u_grid = u(x_grid, epsilon)

            A = -epsilon * utils.reduced_poisson_matrix(x_grid.size, h) + backdiff_matrix(x_grid.size, h)

            # GMRES
            u_hat_grid, iter = gmres(A, f_grid, maxiter=f_grid.size, tol=1e-10)
            if iter == 0: iter = f_grid.size

            results_gmres.append([h, epsilon, iter, l_squared_norm(u_grid, u_hat_grid)])

            # Jacobi
            u_hat_grid, iter, _ = jacobi(A, f_grid, maxiter=f_grid.size, tol=1e-10)
            results_jacobi.append([h, epsilon, iter, l_squared_norm(u_grid, u_hat_grid)])

            # Gauss-Seidel
            u_hat_grid, iter, _ = gauss_seidel(A, f_grid, maxiter=f_grid.size, tol=1e-10)
            results_gauss_seidel.append([h, epsilon, iter, l_squared_norm(u_grid, u_hat_grid)])

    df_gmres = pd.DataFrame(results_gmres, columns=['h', 'epsilon', 'iters', 'l_squared_error'])
    df_gmres.to_csv('sheet_07/df_gmres.csv')
    print(f'{df_gmres=}')
    df_gmres_pivot = df_gmres.pivot(columns='h', index='epsilon', values='iters')
    df_gmres_pivot.to_csv('sheet_07/df_gmres_pivot.csv')
    print(f'{df_gmres_pivot=}')

    df_jacobi = pd.DataFrame(results_jacobi, columns=['h', 'epsilon', 'iters', 'l_squared_error'])
    df_jacobi.to_csv('sheet_07/df_jacobi.csv')
    print(f'{df_jacobi=}')
    df_jacobi_pviot = df_jacobi.pivot(columns='h', index='epsilon', values='iters')
    df_jacobi_pviot.to_csv('sheet_07/df_jacobi_pivot.csv')
    print(f'{df_jacobi_pviot=}')

    df_gauss_seidel = pd.DataFrame(results_gauss_seidel, columns=['h', 'epsilon', 'iters', 'l_squared_error'])
    df_gauss_seidel.to_csv('sheet_07/df_gauss_seidel.csv')
    print(f'{df_gauss_seidel=}')
    df_gauss_seidel_pivot = df_gauss_seidel.pivot(columns='h', index='epsilon', values='iters')
    df_gauss_seidel_pivot.to_csv('sheet_07/df_gauss_seidel_pivot.csv')
    print(f'{df_gauss_seidel_pivot=}')
