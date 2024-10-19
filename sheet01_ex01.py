import numpy as np
import matplotlib.pyplot as plt


### Implementation of explicit Euler

def explicit_euler(ode, y_0, x_grid):
    y_hat_grid = [y_0]
    for i in range(1, x_grid.size):
        h = x_grid[i] - x_grid[i - 1]
        y_hat_grid.append(y_hat_grid[i - 1] + h * ode(x_grid[i - 1], y_hat_grid[i - 1]))

    return np.array(y_hat_grid)


# ODE solved for y'
ode = lambda x, y: (1 / (1 + x) - y) / (1 + x)

# stepsizes
h_1 = .1
h_2 = .2

# Interval discretization
I_1 = np.linspace(0, 1, int(1 / h_1 + 1))
I_2 = np.linspace(0, 1, int(1 / h_2 + 1))

# Compute explicit Euler
y_hat_1 = explicit_euler(ode, 1, I_1)
y_hat_2 = explicit_euler(ode, 1, I_2)

# "True" solution for reference
y = lambda x: (np.log(x + 1) + 1) / (1 + x)
y_grid = y(np.linspace(0, 1, 100))

# plot for plausibility
plt.plot(np.linspace(0, 1, 100), y_grid, 'r', label='true solution')
plt.plot(I_1, y_hat_1, color='b', marker='o', label='h = 0.1')
plt.plot(I_2, y_hat_2, color='g', marker='o', label='h = 0.2')
plt.legend()
plt.show()

# Output values on grid
print(f'{y_hat_1=}')
print(f'{y_hat_2=}')

# Compute errors
error_1 = abs(y_hat_1[-1] - y_grid[-1])
error_2 = abs(y_hat_2[-1] - y_grid[-1])

print(f'error for h = 0.1 is {error_1}, for h = 0.2 is {error_2}.')
