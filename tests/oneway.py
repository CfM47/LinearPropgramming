import numpy as np
from src.simplex import simplex
from src.opt_types import StandardForm, SimplexResult, SimplexTable
from fractions import Fraction
from tests import print_case, evaluate_result

# Test case 1

c = np.array([2, 1, 3, 2, 1], dtype=float)
A = np.array([
  [1, 1, 1, 1, 0], 
  [-1, 1, 2, 0, 1]
  ], dtype=float)
b = np.array([9, 3], dtype=float)
xB = np.array([3, 4], dtype=int)

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])

sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = simplex(sf)

z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(1, np.array([Fraction(x) for x in [3., 6., 0., 0., 0.]]), 12.0, 'Óptimo', x, z, status)

# Test case 2

c = np.array([-3, -1, -2, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 3, 1, 0, 0], 
  [2, 2, 5, 0, 1, 0],
  [4, 1, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([30, 24, 36], dtype=float)
xB = np.array([3, 4, 5], dtype=int)

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])

sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = simplex(sf)

z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(2, np.array([Fraction(x) for x in [8., 4., 0., 18., 0., 0.]]), -28.0, 'Óptimo', x, z, status)

# Test case 3

c = np.array([-1, -1, 0, 0, 0], dtype=float)
A = np.array([
  [4, -1, 1, 0, 0], 
  [2,  1, 0, 1, 0],
  [-5, 2, 0, 0, 1]
  ], dtype=float)
b = np.array([8, 10, 2], dtype=float)
xB = np.array([2, 3, 4], dtype=int)

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])

sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = simplex(sf)

z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(3, np.array([Fraction(x) for x in [2., 6., 6., 0., 0.]]), -8.0, 'Óptimo', x, z, status)

# Test case 4

c = np.array([-18, -12.5, 0, 0, 0], dtype=float)
A = np.array([
  [1, 1, 1, 0, 0], 
  [1, 0, 0, 1, 0],
  [0, 1, 0, 0, 1]
  ], dtype=float)
b = np.array([20., 12. , 16.], dtype=float)
xB = np.array([2, 3, 4], dtype=int)

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])

sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = simplex(sf)

z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(4, np.array([Fraction(x) for x in [12., 8., 0., 0., 8.]]), -316.0, 'Óptimo', x, z, status)

# Test case 5

c = np.array([-5, 3, 0, 0], dtype=float)
A = np.array([
  [1, -1, 1, 0], 
  [2, 1, 0, 1]
  ], dtype=float)
b = np.array([1, 2], dtype=float)
xB = np.array([2, 3], dtype=int)

c = np.array([Fraction(x) for x in c])
A = np.array([[Fraction(x) for x in row] for row in A])
b = np.array([Fraction(x) for x in b])

sf = StandardForm(c, A, b, xB, len(c) - len(xB), len(xB))

result = simplex(sf)

z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(5, np.array([Fraction(x) for x in [1., 0., 0., 0.]]), -5.0, 'Óptimo', x, z, status)