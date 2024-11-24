import numpy as np
import scipy.sparse as sp
import utils
import matplotlib.pyplot as plt

from numpy import exp
from sheet02_ex03 import l_squared_norm

# This code answers Problems 06.03 c) and 06.03 d) concurrently.

# set sparse matrix corresponding to central difference
def centraldiff_matrix(n: int, h: float) -> sp.csc_matrix:
    offdiag = np.ones((n,))
    diag = np.zeros((n,))
    data = np.array([-offdiag, diag, offdiag])
    offset = np.array([-1, 0, 1])

    return 1/(2*h) * sp.dia_matrix((data, offset), shape=(n, n)).tocsc()


if __name__ == "__main__":
    u = lambda x, epsilon: x - (exp(-(1-x)/epsilon) - exp(-1/epsilon))/(1 - exp(-1/epsilon))
    stepsize = 1/128

    epsilons = np.array([10**(-2*i) for i in range(4)])

    fig, ax = plt.subplots(2,2, layout="constrained")
    for k in range(epsilons.size):
        # row and col indices for subplots
        i = k // 2
        j = k % 2

        epsilon = epsilons[k]

        x_grid = utils.grid_from_stepsize(0, 1, stepsize)
        f_grid = np.ones(x_grid.size)
        u_grid = u(x_grid, epsilon)

        A = -epsilon * utils.reduced_poisson_matrix(x_grid.size, stepsize) + centraldiff_matrix(x_grid.size, stepsize)

        u_hat_grid = sp.linalg.spsolve(A, f_grid)
        ax[i,j].plot(x_grid, u_hat_grid, color='r', label=r'$\hat u$')
        ax[i, j].plot(x_grid, u_grid, color='b', label=r'$u$')
        ax[i,j].set_title(r'$\varepsilon=$' + str(epsilon))

        print(f'l-squared norm is {l_squared_norm(u_grid, u_hat_grid)}')

    ax[1,1].legend()
    fig.suptitle('Sheet 6, problems 3c-d')
    fig.savefig('sheet_06/fig.png')
