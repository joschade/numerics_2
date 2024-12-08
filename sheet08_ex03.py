import numpy as np

import utils
import pandas as pd

from scipy.sparse.linalg import gmres

from numpy import exp
from sheet02_ex03 import l_squared_norm
from sheet07_ex03 import backdiff_matrix


u = lambda x, epsilon: x - (exp(-(1-x)/epsilon) - exp(-1/epsilon))/(1 - exp(-1/epsilon))
hs = np.array([1/2**k for k in range(3,9)])
epsilons = np.array([10**(-2*i) for i in range(4)])
restarts = [5, 10, 20, 30, 40, 50]

class gmres_counter:
    def __init__(self):
        self.niter = 0
    def call(self, x):
        self.niter += 1

# This section answers 08.03
if __name__ == "__main__":
    print('Sheet 08 Question 03')

    tol = 1e-3

    results_gmres = []
    for epsilon in epsilons:
        for h in hs:
            for restart in restarts:
                x_grid = utils.grid_from_stepsize(0, 1, h)
                f_grid = np.ones(x_grid.size)
                u_grid = u(x_grid, epsilon)

                A = -epsilon * utils.reduced_poisson_matrix(x_grid.size, h) + backdiff_matrix(x_grid.size, h)

                counter = gmres_counter()

                # GMRES
                u_hat_grid, info = gmres(A, f_grid, maxiter=1000, tol=tol, callback = counter.call, x0 = np.zeros(x_grid.shape))

                iter = counter.niter

                results_gmres.append([h, epsilon, counter.niter, restart, l_squared_norm(u_grid, u_hat_grid)])


    df_gmres = pd.DataFrame(results_gmres, columns=['h', 'epsilon', 'iters', 'restart', 'l_squared_error'])
    df_gmres.to_csv('sheet_08/df_gmres.csv')
    print(f'{df_gmres=}')
    df_gmres_pivot = df_gmres.pivot(columns=['h', 'epsilon'], index='restart', values='iters')
    df_gmres_pivot.to_csv('sheet_08/df_gmres_pivot.csv')
    print(f'{df_gmres_pivot=}')
