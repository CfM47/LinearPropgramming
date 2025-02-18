import numpy as np
from src.dual_simplex import dual_simplex, is_dual_feasible
from tests import print_case, evaluate_result

# Test case 1  

c = np.array([3, 2, 1, 0, 0])
A = np.array([
  [-1, -1, 1, 1, 0],
  [2, 1, -2, 0, 1]
])
b = np.array([-5, -4])
xB = np.array([3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(1, np.array([0, 14., 9., 0, 0]), 37, 'Óptimo', r, z, status)

# Test case 2

c = np.array([3, 1, 2, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 3, 1, 0, 0], 
  [2, 2, 5, 0, 1, 0],
  [4, 1, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([-30, 24, 36], dtype=float)
xB = np.array([3, 4, 5])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(2, None, None, 'Infactible', r, z, status)

# Test case 3

c = np.array([1, 1, 0, 0, 0], dtype=float)
A = np.array([
  [4, -1, 1, 0, 0], 
  [2,  1, 0, 1, 0],
  [-5, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([8, -10, 2], dtype=float)
xB = np.array([2, 3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(3, None, None, 'Infactible', r, z, status)

# Test case 4

c = np.array([18, 12.5, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 1, 0, 0], 
  [1, 0, 0, 1, 0],
  [0, 1, 0, 0, 1]
  ], dtype=float)
b = np.array([20., -12. , -16.], dtype=float)
xB = np.array([2, 3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(4, None, None, 'Infactible', r, z, status)

# Test case 5

c = np.array([5, 3, 0, 0], dtype=float)
A = np.array([
  [1, -1, 1, 0], 
  [2, 1, 0, 1]
  ], dtype=float)
b = np.array([1, -2], dtype=float)
xB = np.array([2, 3])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(5, None, None, 'Infactible', r, z, status)

# Test case 6

c = np.array([11, 1, 3, 0, 0], dtype=float)
A = np.array([
  [-2, -7.5, -3, 1, 0], 
  [-20, 5, 10, 0, 1]
  ], dtype=float)
b = np.array([-10000, -30000], dtype=float)
xB = np.array([3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(6, np.array([6875/4, 875, 0, 0, 0]), 79125/4, 'Óptimo', r, z, status)


# Test case 7

c = np.array([2, 1, 0, 0], dtype=float)
A = np.array([
  [2, -1, 1, 0],
  [1, -5, 0, 1]
  ], dtype=float)
b = np.array([2, -4], dtype=float)
xB = np.array([2, 3])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(7, np.array([0, 4/5, 14/5, 0]), 4/5, 'Óptimo', r, z, status)

# Test case 8

c = np.array([1, 3, 0, 0, 0], dtype=float)
A = np.array([
  [1, -1, 1, 0, 0],
  [-1, -1, 0, 1, 0],
  [-1, 4, 0, 0, 1]
  ], dtype=float)
b = np.array([8, -3, 2], dtype=float)
xB = np.array([2, 3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(8, np.array([3, 0, 5, 0, 5]), 3, 'Óptimo', r, z, status)

# Test 9

c = np.array([1, 2, 0, 0, 0], dtype=float)
A = np.array([
  [1, 2, 1, 0, 0],
  [-2, -6, 0, 1, 0],
  [0, 1, 0, 0, 1]
  ])
b = np.array([4, -12, 1], dtype=float)
xB = np.array([2, 3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(9, None, None, 'Infactible', r, z, status)

# Test 10

c = np.array([1, 3, 0, 0, 0], dtype=float)
A = np.array([
  [-1, 1, 1, 0, 0],
  [-1, -1, 0, 1, 0],
  [-1, 4, 0, 0, 1]
  ])
b = np.array([1, -3, 2], dtype=float)
xB = np.array([2, 3, 4])

r, z, _, _, _, status = dual_simplex(c, A, b, xB)
evaluate_result(10, np.array([3, 0, 4, 0, 5]), 3, 'Óptimo', r, z, status)