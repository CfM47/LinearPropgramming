import numpy as np
from src.dual_simplex import dual_simplex
from src.opt_types import StandardForm
from fractions import Fraction
from tests import print_case, evaluate_result

# Test case 1  

c = np.array([3, 2, 1, 0, 0])
A = np.array([
  [-1, -1, 1, 1, 0],
  [2, 1, -2, 0, 1]
])
b = np.array([-5, -4])
xB = np.array([3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(1, np.array([Fraction(x) for x in [0, 14., 9., 0, 0]]), Fraction(37), 'Óptimo', x, z, status)

# Test case 2

c = np.array([3, 1, 2, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 3, 1, 0, 0], 
  [2, 2, 5, 0, 1, 0],
  [4, 1, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([-30, 24, 36], dtype=float)
xB = np.array([3, 4, 5])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(2, None, None, 'Infactible', x, z, status)

# Test case 3
c = np.array([1, 1, 0, 0, 0], dtype=float)
A = np.array([
  [4, -1, 1, 0, 0], 
  [2,  1, 0, 1, 0],
  [-5, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([8, -10, 2], dtype=float)
xB = np.array([2, 3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(3, None, None, 'Infactible', x, z, status)

# Test case 4
c = np.array([18, 12.5, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 1, 0, 0], 
  [1, 0, 0, 1, 0],
  [0, 1, 0, 0, 1]
  ], dtype=float)
b = np.array([20., -12. , -16.], dtype=float)
xB = np.array([2, 3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(4, None, None, 'Infactible', x, z, status)

# Test case 5

c = np.array([5, 3, 0, 0], dtype=float)
A = np.array([
  [1, -1, 1, 0], 
  [2, 1, 0, 1]
  ], dtype=float)
b = np.array([1, -2], dtype=float)
xB = np.array([2, 3])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(5, None, None, 'Infactible', x, z, status)

# Test case 6

c = np.array([11, 1, 3, 0, 0], dtype=float)
A = np.array([
  [-2, -7.5, -3, 1, 0], 
  [-20, 5, 10, 0, 1]
  ], dtype=float)
b = np.array([-10000, -30000], dtype=float)
xB = np.array([3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(6, np.array([Fraction(6875, 4), Fraction(875), Fraction(0), Fraction(0), Fraction(0)]), Fraction(79125, 4), 'Óptimo', x, z, status)

# Test case 7
c = np.array([2, 1, 0, 0], dtype=float)
A = np.array([
  [2, -1, 1, 0],
  [1, -5, 0, 1]
  ], dtype=float)
b = np.array([2, -4], dtype=float)
xB = np.array([2, 3])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(7, np.array([Fraction(0), Fraction(4, 5), Fraction(14, 5), Fraction(0)]), Fraction(4, 5), 'Óptimo', x, z, status)

# Test case 8
c = np.array([1, 3, 0, 0, 0], dtype=float)
A = np.array([
  [1, -1, 1, 0, 0],
  [-1, -1, 0, 1, 0],
  [-1, 4, 0, 0, 1]
  ], dtype=float)
b = np.array([8, -3, 2], dtype=float)
xB = np.array([2, 3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(8, np.array([Fraction(3), Fraction(0), Fraction(5), Fraction(0), Fraction(5)]), Fraction(3), 'Óptimo', x, z, status)

# Test case 9
c = np.array([1, 2, 0, 0, 0], dtype=float)
A = np.array([
  [1, 2, 1, 0, 0],
  [-2, -6, 0, 1, 0],
  [0, 1, 0, 0, 1]
  ])
b = np.array([4, -12, 1], dtype=float)
xB = np.array([2, 3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(9, None, None, 'Infactible', x, z, status)

# Test case 10
c = np.array([1, 3, 0, 0, 0], dtype=float)
A = np.array([
  [-1, 1, 1, 0, 0],
  [-1, -1, 0, 1, 0],
  [-1, 4, 0, 0, 1]
  ])
b = np.array([1, -3, 2], dtype=float)
xB = np.array([2, 3, 4])

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])
sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = dual_simplex(sf)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(10, np.array([Fraction(3), Fraction(0), Fraction(4), Fraction(0), Fraction(5)]), Fraction(3), 'Óptimo', x, z, status)