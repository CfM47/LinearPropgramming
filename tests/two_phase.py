import numpy as np
from src.simplex_twophase import simplex_twophase
from tests import print_case, evaluate_result

# Test case 2

c = np.array([-3, -1, -2], dtype=float)
A = np.array([
  [1, 1, 3], 
  [2, 2, 5],
  [4, 1, 2]
  ], dtype=float)
b = np.array([30, 24, 36], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)

evaluate_result(2, np.array([8., 4., 0., 18., 0., 0.]), -28.0, 'Óptimo', r, z, status)

# Test case 3

c = np.array([-1, -1], dtype=float)
A = np.array([
  [4, -1], 
  [2,  1],
  [-5, 2]
  ], dtype=float)
b = np.array([8, 10, 2], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)

evaluate_result(3, np.array([2., 6., 6., 0., 0.]), -8.0, 'Óptimo', r, z, status)

# Test case 4

c = np.array([-18, -12.5], dtype=float)
A = np.array([
  [1, 1], 
  [1, 0],
  [0, 1]
  ], dtype=float)
b = np.array([20., 12. , 16.], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)

evaluate_result(4, np.array([12., 8., 0., 0., 8.]), -316.0, 'Óptimo', r, z, status)

# Test case 5

c = np.array([-5, 3], dtype=float)
A = np.array([
  [1, -1], 
  [2, 1]
  ], dtype=float)
b = np.array([1, 2], dtype=float)
r, z, y0, status = simplex_twophase(c, A, b)

evaluate_result(5, np.array([1., 0., 0., 0.]), -5.0, 'Óptimo', r, z, status)

# Test case 6

c = np.array([-1, -1, -3], dtype=float)
A = np.array([
  [-2, -7.5, -3], 
  [-20, 5, 10]
  ], dtype=float)
b = np.array([-10000, -30000], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)

evaluate_result(6, None, None, 'Problema no acotado', r, z, status)

# Test case 7

c = np.array([-2, 1], dtype=float)
A = np.array([
  [2, -1],
  [1, -5]
  ], dtype=float)
b = np.array([2, -4], dtype=float)

r, z , y0, status = simplex_twophase(c, A, b)

evaluate_result(7, np.array([14/9, 10/9, 0, 0]), -2., "Óptimo", r, z, status)

# Test case 8

c = np.array([4, 1, 1], dtype=float)
A = np.empty((0, 3), dtype=float)
b = np.empty(0, dtype=float)
E = np.array([
  [2, 1, 2],
  [3, 3, 1]
  ], dtype=float)
d = np.array([4, 3], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b, E, d)

evaluate_result(8, np.array([0, 2/5, 9/5]), 11/5, 'Óptimo', r, z, status)

c = np.array([-1, -3], dtype=float)
A = np.array([
  [1, -1],
  [-1, -1],
  [-1, 4]
  ], dtype=float)
b = np.array([8, -3, 2], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)

evaluate_result(9, np.array([34/3, 10/3, 0, 35/3, 0]), -64/3, 'Óptimo', r, z, status)

# Test 10

c = np.array([-1, 2], dtype=float)
A = np.array([
  [1, 2],
  [-2, -6],
  [0, 1]
  ])
b = np.array([4, -12, 1], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)
evaluate_result(10, None, None, 'Infactible', r, z, status)

c = np.array([-1, -3], dtype=float)
A = np.array([
  [-1, 1],
  [-1, -1],
  [-1, 4]
  ])
b = np.array([1, -3, 2], dtype=float)

r, z, y0, status = simplex_twophase(c, A, b)
evaluate_result(11, None, None, 'Problema no acotado', r, z, status)