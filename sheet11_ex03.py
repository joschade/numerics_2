import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv, spilu, spsolve
from utils import grid_from_stepsize, reduced_poisson_matrix
from sheet02_ex03 import f

def dlu_decomp(A: sp.csc_matrix):
    return sp.dia_matrix((A.diagonal(0), 0), shape=A.shape).tocsc(), sp.tril(A, k=-1), sp.triu(A, k=1)

def pcg(A: sp.csc_matrix, b: np.array, M: sp.csc_matrix, x_0: np.array, eps: float):
    r_old = b - A.dot(x_0)
    z_old = spsolve(M, r_old)
    p = z_old
    k = 0
    x = x_0

    # abs added to catch the non-defined case for <z_old, r_old> < 0
    while abs(np.dot(z_old, r_old))**.5 > eps:
        k = k+1
        s = A.dot(p)
        nu = np.dot(z_old, r_old)/np.dot(p, s)
        x = x + nu * p
        r_new = r_old - nu * s
        z_new = spsolve(M, r_new)
        mu = np.dot(r_new, z_new)/np.dot(r_old, z_old)
        p = z_new + mu * p

        r_old = r_new
        z_old = z_new

    return x, k

hs = [1/2**k for k in range(3,11)]

for h in hs:

    x_grid = grid_from_stepsize(0,1,h)[1:-1]
    f_grid = f(x_grid)


    A = reduced_poisson_matrix(x_grid.size, h)
    D, L, U = dlu_decomp(A)
    L_ichol = spilu(A).L

    x_0 = np.zeros(x_grid.size)

    Ms = [sp.identity(A.shape[0]), D, (D+L).dot(inv(D)).dot((D+L).transpose()), L_ichol.transpose().dot(L_ichol)]
    M_names = ['noPrecon', 'Jacobi', 'SSOR', 'iCholesky']

    for i in range(len(Ms)):
        sol, iter = pcg(A, f_grid, Ms[i], x_0, 10e-10)

        print(f' for {h=} and conditioning by {M_names[i]}: iterations={iter}')

