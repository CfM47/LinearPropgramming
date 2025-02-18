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

r, z, y0, status = simplex(c, A, b, xB)

evaluate_result(1, np.array([3., 6., 0., 0., 0.]), 12.0, 'Óptimo', r, z, status)

# Test case 2

c = np.array([-3, -1, -2, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 3, 1, 0, 0], 
  [2, 2, 5, 0, 1, 0],
  [4, 1, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([30, 24, 36], dtype=float)
xB = np.array([3, 4, 5], dtype=int)

r, z, y0, status = simplex(c, A, b, xB)

evaluate_result(2, np.array([8., 4., 0., 18., 0., 0.]), -28.0, 'Óptimo', r, z, status)

# Test case 3

c = np.array([-1, -1, 0, 0, 0], dtype=float)
A = np.array([
  [4, -1, 1, 0, 0], 
  [2,  1, 0, 1, 0],
  [-5, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([8, 10, 2], dtype=float)
xB = np.array([2, 3, 4], dtype=int)

r, z, y0, status = simplex(c, A, b, xB)

evaluate_result(3, np.array([2., 6., 6., 0., 0.]), -8.0, 'Óptimo', r, z, status)

# Test case 4

c = np.array([-18, -12.5, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 1, 0, 0], 
  [1, 0, 0, 1, 0],
  [0, 1, 0, 0, 1]
  ], dtype=float)
b = np.array([20., 12. , 16.], dtype=float)
xB = np.array([2, 3, 4], dtype=int)

r, z, y0, status = simplex(c, A, b, xB)

evaluate_result(4, np.array([12., 8., 0., 0., 8.]), -316.0, 'Óptimo', r, z, status)

# Test case 5

c = np.array([-5, 3, 0, 0], dtype=float)
A = np.array([
  [1, -1, 1, 0], 
  [2, 1, 0, 1]
  ], dtype=float)
b = np.array([1, 2], dtype=float)
xB = np.array([2, 3], dtype=int)

r, z, y0, status = simplex(c, A, b, xB)

evaluate_result(5, np.array([1., 0., 0., 0.]), -5.0, 'Óptimo', r, z, status)