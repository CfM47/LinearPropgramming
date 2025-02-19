import numpy as np
from src.gomory import gomory 
from src.opt_types import LinearProblem, SimplexResult, SimplexTable
from fractions import Fraction
from tests import print_case, evaluate_result

# Test case 1

c = np.array([1, 1])
A = np.array([
  [-2, 2],
  [2, -5],
  [5, 3]])
b = np.array([-1, 0, 30])
E = np.empty((0,2))
d = np.empty(0)

lp = LinearProblem(c, A, b, E, d)

result = gomory(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(1, np.array([Fraction(x) for x in [2, 1, 1, 1, 17, 0, 0]]), 3, 'Óptimo', x, z, status)

# Test case 2

c = np.array([-1, -1])
A = np.array([
  [3, 2],
  [0, 1],
  ])
b = np.array([5, 2])
E = np.empty((0,2))
d = np.empty(0)

lp = LinearProblem(c, A, b, E, d)

result = gomory(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(2, np.array([Fraction(x) for x in [0, 2]]), -2, 'Óptimo', x[range(2)], z, status)

# Test case 3

c = np.array([-1, -4])
A = np.array([
  [2, 4],
  [5, 3],
  ])
b = np.array([7, 15])
E = np.empty((0,2))
d = np.empty(0)

lp = LinearProblem(c, A, b, E, d)

result = gomory(lp)
z = result.table.z if result.table is not None else None
x, status = result.x, result.status

evaluate_result(3, np.array([Fraction(x) for x in [1, 1]]), -5, 'Óptimo', x[range(2)], z, status)
