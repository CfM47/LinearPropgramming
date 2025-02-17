import numpy as np
from src.simplex import simplex
from tests import print_case, evaluate_result

# Test case 1

c = np.array([2, 1, 3, 2, 1], dtype=float)
A = np.array([
  [1, 1, 1, 1, 0], 
  [-1, 1, 2, 0, 1]
  ], dtype=float)
b = np.array([9, 3], dtype=float)
xB = np.array([3, 4], dtype=int)

r, z, status = simplex(c, A, b, xB)

print_case(1, 'Problema de optimización lineal en la forma estándar')
evaluate_result(1, np.array([3., 6., 0., 0., 0.]), 12.0, 'Óptimo', r, z, status)

# Test case 2


