import numpy as np
import scipy.sparse as sp
from jeepney.low_level import Array
from pandas.core.arrays.arrow import ListAccessor


def reduced_poisson_matrix(n: int, h=1.0) -> sp.dia_matrix:
    offdiag = np.ones((n,))
    diag = -2 * offdiag
    data = np.array([offdiag, diag, offdiag])
    offset = np.array([-1, 0, 1])

    return 1/h**2*sp.dia_matrix((data, offset), shape=(n, n)).tocsc()


def explicit_method(ode: callable, y_0: float, x_grid: np.array, Phi: callable) -> np.array:
    y_hat_grid = [y_0]
    for i in range(1, x_grid.size):
        h = x_grid[i] - x_grid[i - 1]
        y_hat_grid.append(y_hat_grid[i - 1] + h * Phi(ode, x_grid[i - 1], y_hat_grid[i - 1]))

    return np.array(y_hat_grid)

def grid_from_stepsize(start=0., stop=1., stepsize=.1) -> np.array:
    return np.linspace(start=start, stop=stop, num=int(np.ceil((stop-start)/stepsize)+1))

def cartesian_product(array_1: np.array, array_2: np.array) -> np.array:
    return np.repeat(array_1, array_2.size), np.tile(array_2, array_1.size)

