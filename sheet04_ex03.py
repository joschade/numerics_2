import numpy as np
import pandas as pd
from utils import reduced_poisson_matrix
from sheet02_ex03 import f
from sheet03_ex03 import sor
from concurrent.futures import ProcessPoolExecutor

optimal_relax = lambda h: 2/(1+np.sin(h*np.pi))


if __name__ == "__main__":
    # define list of inverse stepsizes
    hs = [2 ** i for i in range(3, 9)]

    # for parallelization
    def sor_mp(h: float):
        x_grid = np.linspace(0, 1, h)[1:-1]
        f_grid = f(x_grid)

        A = reduced_poisson_matrix(x_grid.size)
        x_0 = np.zeros(x_grid.size)
        _, iter, err = sor(A, x_0, f_grid, optimal_relax(1 / h))
        return h, optimal_relax(1 / h), iter, err

    with ProcessPoolExecutor(max_workers=len(hs)) as executor:
        results = list(executor.map(sor_mp, hs))

    df_sor = pd.DataFrame(results, columns=['stepsize_inv', 'omega', 'iterations', 'error'])
    df_sor.to_csv('sheet_04/df_sor.csv')
    print(df_sor)