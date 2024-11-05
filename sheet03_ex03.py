import scipy.sparse as sp
import numpy as np
import pandas as pd
from utils import reduced_poisson_matrix, cartesian_product, grid_from_stepsize
from scipy.sparse.linalg import norm, inv
from sheet02_ex03 import f

# for multiprocessing
import concurrent.futures


if __name__ == "__main__":
    print('Sheet 03 Question 03 Subproblem a)')

    # define list of inverse stepsizes
    hs = [2 ** i for i in range(3, 9)]


# subproblem a)
for h in hs:
    A = 1 / h ** 2 * reduced_poisson_matrix(h)
    cond = norm(A, ord=1) * norm(inv(A), ord=1)
    print(f'condition number stepsize 1/{h} is {cond}')

# subproblem b)
def damped_jacobi(A: np.array, x_0: np.array, b:np.array, omega: float, minerr=1e-10, maxiter=1e5) -> np.array:
    D = sp.dia_matrix((A.diagonal(0), 0), shape=A.shape).tocsc()

    def update(x_k):
        return x_k + omega * inv(D).dot((b-A.dot(x_k)))

    iter = 0
    err = minerr
    x_k = x_0
    while err >= minerr and iter <= maxiter:
        x_k = update(x_k)
        err = np.linalg.norm(b - A.dot(x_k))
        iter = iter + 1

    return x_k, iter-1, err

if __name__ == "__main__":
    print('Sheet 03 Question 03 Subproblem b)')

    list_dampedjac = []
    omegas = np.linspace(.1,1.1, 11)

    ## boil down to arguments to be iterated over
    def damped_jacobi_mp(h, omega: float):
        x_grid = np.linspace(0, 1, h)[1:-1]
        f_grid = f(x_grid)

        A = reduced_poisson_matrix(x_grid.size)
        x_0 = np.zeros(x_grid.size)
        _, iter, err = damped_jacobi(A, x_0, f_grid, omega)
        return h, omega, iter, err



    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(damped_jacobi_mp, *cartesian_product(np.array(hs), omegas)))

    df_dampedjac = pd.DataFrame(results, columns=['stepsize_inv', 'omega', 'iterations', 'error'])
    df_dampedjac.to_csv('/sheet_03/df_dampedjac.csv')
    print(df_dampedjac)

# subproblem c)
def sor(A: np.array, x_0: np.array, b:np.array, omega: float, minerr=1e-10, maxiter=1e5) -> np.array:
    D = sp.dia_matrix((A.diagonal(0), 0), shape=A.shape).tocsc()
    L = sp.tril(A, k=-1)

    def update(x_k):
        return x_k + inv(D/omega+L).dot((b-A.dot(x_k)))

    iter = 0
    err = minerr
    x_k = x_0
    while err >= minerr and iter <= maxiter:
        x_k = update(x_k)
        err = np.linalg.norm(b - A.dot(x_k))
        iter = iter + 1

    return x_k, iter-1, err

if __name__ == "__main__":
    print('Sheet 03 Question 03 Subproblem c)')

    epsilons = grid_from_stepsize(.1, 2., .1)


    def sor_mp(h, omega: float):
        x_grid = np.linspace(0, 1, h)[1:-1]
        f_grid = f(x_grid)

        A = reduced_poisson_matrix(x_grid.size)
        x_0 = np.zeros(x_grid.size)
        _, iter, err = sor(A, x_0, f_grid, omega)
        return h, omega, iter, err


        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(sor_mp, *cartesian_product(np.array(hs), omegas)))

        df_sor = pd.DataFrame(results, columns=['stepsize_inv', 'omega', 'iterations', 'error'])
        df_sor.to_csv('/sheet_03/df_sor.csv')
        print(df_dampedjac)
