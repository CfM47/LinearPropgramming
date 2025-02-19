import numpy as np
from src.simplex_twophase import simplex_twophase
from src.opt_types import LinearProblem, SimplexResult, SimplexTable
from fractions import Fraction
from tests import print_case, evaluate_result

# Test case 2

c = np.array([-3, -1, -2], dtype=float)
A = np.array([
  [1, 1, 3], 
  [2, 2, 5],
  [4, 1, 2]
  ], dtype=float)
b = np.array([30, 24, 36], dtype=float)
E = np.zeros((0,3))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(2, np.array([Fraction(x) for x in [8., 4., 0., 18., 0., 0.]]), -28.0, 'Óptimo', x, z, status)

# Test case 3

c = np.array([-1, -1], dtype=float)
A = np.array([
  [4, -1], 
  [2,  1],
  [-5, 2]
  ], dtype=float)
b = np.array([8, 10, 2], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(3, np.array([Fraction(x) for x in [2., 6., 6., 0., 0.]]), -8.0, 'Óptimo', x, z, status)

# Test case 4

c = np.array([-18, -12.5], dtype=float)
A = np.array([
  [1, 1], 
  [1, 0],
  [0, 1]
  ], dtype=float)
b = np.array([20., 12. , 16.], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(4, np.array([Fraction(x) for x in [12., 8., 0., 0., 8.]]), -316.0, 'Óptimo', x, z, status)

# Test case 5

c = np.array([-5, 3], dtype=float)
A = np.array([
  [1, -1], 
  [2, 1]
  ], dtype=float)
b = np.array([1, 2], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(5, np.array([Fraction(x) for x in [1., 0., 0., 0.]]), -5.0, 'Óptimo', x, z, status)

# Test case 6

c = np.array([-1, -1, -3], dtype=float)
A = np.array([
  [-2, -7.5, -3], 
  [-20, 5, 10]
  ], dtype=float)
b = np.array([-10000, -30000], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(6, None, None, 'Problema no acotado', x, z, status)

# Test case 7

c = np.array([-2, 1], dtype=float)
A = np.array([
  [2, -1],
  [1, -5]
  ], dtype=float)
b = np.array([2, -4], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(7, np.array([Fraction(14, 9), Fraction(10, 9), Fraction(0), Fraction(0)]), -2., "Óptimo", x, z, status)

# Test case 8

c = np.array([4, 1, 1], dtype=float)
A = np.empty((0, 3), dtype=float)
b = np.empty(0, dtype=float)
E = np.array([
  [2, 1, 2],
  [3, 3, 1]
  ], dtype=float)
d = np.array([4, 3], dtype=float)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(8, np.array([Fraction(0), Fraction(2, 5), Fraction(9, 5), 0, 0, 0, 0]), Fraction(11, 5), 'Óptimo', x, z, status)

# Test case 9

c = np.array([-1, -3], dtype=float)
A = np.array([
  [1, -1],
  [-1, -1],
  [-1, 4]
  ], dtype=float)
b = np.array([8, -3, 2], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(9, np.array([Fraction(34, 3), Fraction(10, 3), Fraction(0), Fraction(35, 3), Fraction(0)]), Fraction(-64, 3), 'Óptimo', x, z, status)

# Test 10

c = np.array([-1, 2], dtype=float)
A = np.array([
  [1, 2],
  [-2, -6],
  [0, 1]
  ])
b = np.array([4, -12, 1], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(10, None, None, 'Infactible', x, z, status)

c = np.array([-1, -3], dtype=float)
A = np.array([
  [-1, 1],
  [-1, -1],
  [-1, 4]
  ])
b = np.array([1, -3, 2], dtype=float)
E = np.zeros((0, A.shape[1]))
d = np.zeros(0)

lp = LinearProblem(c, A, b, E, d)

result = simplex_twophase(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(11, None, None, 'Problema no acotado', x, z, status)