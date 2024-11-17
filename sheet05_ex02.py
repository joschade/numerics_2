import numpy as np
import pandas as pd
import scipy as sp
from utils import reduced_poisson_matrix
from sheet02_ex03 import f
from concurrent.futures import ProcessPoolExecutor

def richardson(A: np.array, x_0: np.array, b:np.array, alpha: float, minerr=1e-10, maxiter=1e5) -> np.array:
    I = sp.sparse.identity(n=A.shape[0])

    def update(x_k):
        return (I - .9 * alpha * A).dot(x_k) + .9 * alpha * b

    iter = 0
    err = minerr
    x_k = x_0
    while err >= minerr and iter <= maxiter:
        x_k = update(x_k)
        err = np.linalg.norm(b - A.dot(x_k))
        iter = iter + 1

    return x_k, iter-1, err

def optimal_relax(h: float, n: int) -> float:
    # initalize minimal and maximal eigenvalue of a spd matrix
    min = np.inf
    max = -np.inf

    for k in range(1,n+1):
        eigval = 4/(h**2) * np.sin(k*np.pi/(2*(n+1)))**2
        if eigval < min: min = eigval
        elif eigval > max: max = eigval

    return 2/(min + max)

if __name__ == "__main__":
    # define list of inverse stepsizes
    hs = [2 ** i for i in range(3, 9)]


    # for parallelization
    def richardson_mp(h: float):
        x_grid = np.linspace(0, 1, h)[1:-1]
        f_grid = f(x_grid)

        A = -1/h**2 * reduced_poisson_matrix(x_grid.size)
        x_0 = np.zeros(x_grid.size)
        _, iter, err = richardson(A, x_0, f_grid, optimal_relax(1 / h, x_grid.size))
        return h, optimal_relax(1 / h, x_grid.size), iter, err

    with ProcessPoolExecutor(max_workers=len(hs)) as executor:
        results = list(executor.map(richardson_mp, hs))

    df_richardson = pd.DataFrame(results, columns=['stepsize_inv', 'alpha', 'iterations', 'error'])
    df_richardson.to_csv('sheet_05/df_richardson.csv')
    print(df_richardson)
