
import numpy as np
from solution_methods import Jacobi

A = np.array([
    [2, -2],
    [-2, 2]
])

b = np.array([0, 0])
vec = np.array( [5, 2])

x, residual, iterations = Jacobi(A, b, vec)
print(f'solution: {x}')
print(f'residual: {residual}')
print(f'iterations needed: {iterations}')