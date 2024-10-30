import numpy as np
from numpy import pi, sin, cos, sqrt
from utils import reduced_poisson_matrix, grid_from_stepsize
from scipy.sparse.linalg import spsolve

f = lambda x: - 6*pi*cos(3*pi*x) + 9*pi**2*x*sin(3*pi*x)
u = lambda x: - x*sin(3*pi*x)
stepsizes = [1/2**i for i in range(3,9)]

def l_squared_norm(f_grid, f_hat_grid: np.array) -> float:
    assert f_grid.size == f_hat_grid.size, "f_grid and f_hat_grid need to be of same size"
    squared_sum = np.sum((f_grid - f_hat_grid)**2)

    # note that the outer nodes are already removed, so f_grid.size is the number of inner nodes
    return sqrt(1/(f_grid.size)*squared_sum)


if __name__ == "__main__":
    for stepsize in stepsizes:
        # consider only internal nodes
        grid = grid_from_stepsize(stepsize=stepsize)[1:-1]

        f_grid = f(grid)[1:-1]
        u_grid = u(grid)

        pois_matrix = reduced_poisson_matrix(f_grid.size)
        u_hat_grid = spsolve(pois_matrix, stepsize**2*f_grid)

        norm = l_squared_norm(u_grid, u_hat_grid)

        print(f'for {stepsize=}, {norm=}')