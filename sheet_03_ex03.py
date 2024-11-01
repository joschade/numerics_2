import scipy.sparse as sp
import numpy as np
from utils import reduced_poisson_matrix, grid_from_stepsize
from scipy.sparse.linalg import norm, inv
from sheet02_ex03 import f



# define list of stepsizes

if __name__ == "__main__":
    print('Sheet 03 Question 03 Subproblem a)')
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

if __name__ == "__main__": """
    print('Sheet 03 Question 03 Subproblem b)')

    epsilons = np.linspace(.1,1.1, 11)

    for h in hs:
        print(f'for {h=}:')
        x_grid = np.linspace(0, 1, h)[1:-1]
        f_grid = f(x_grid)

        A = reduced_poisson_matrix(x_grid.size)
        x_0 = np.zeros(x_grid.size)
        for epsilon in epsilons:
            print(f'  for {epsilon=}:')
            x, iter, err = damped_jacobi(A, x_0, f_grid, epsilon)
            print(f'    iterations: {iter}, error: {err}') """

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
    print('Sheet 03 Question 03 Subproblem b)')

    epsilons = grid_from_stepsize(.1, 2., .1)


    for h in hs:
        print(f'for {h=}:')
        x_grid = np.linspace(0, 1, h)[1:-1]
        f_grid = f(x_grid)

        A = reduced_poisson_matrix(x_grid.size)
        x_0 = np.zeros(x_grid.size)
        relax = []
        for epsilon in epsilons:
            print(f'  for {epsilon=}:')

            x, iter, err = sor(A, x_0, f_grid, epsilon)
            print(f'    iterations: {iter}, error: {err}')
            relax.append(iter)

        argmax = np.argmin(np.array(relax))
        print(f'  best relaxation factor: {epsilons[argmax]}')
