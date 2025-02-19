import numpy as np
from src.full_simplex import full_simplex
from src.opt_types import StandardForm, SimplexResult, SimplexTable, LinearProblem
from src.utils import standarize, rationalize, make_integer
from src.post_optimization import add_restriction
from fractions import Fraction

def all_integer(vector: np.ndarray[Fraction]):
  return np.all([x.is_integer() for x in vector])

def frac(x: Fraction):
  floor = np.floor(float(x))
  return x - Fraction.from_float(floor)
    

def gomory(
  lp: LinearProblem,
  print_it: bool = False
):
  """
  Resuelve un problema de optimización de la forma
  min c^T x
  s.a Ax <= b
    x>=0, x \in Z 
  """
  
  lp = rationalize(lp)
  lp = make_integer(lp)
  sf = standarize(lp)
  
  # resolver el problema continuo asociado
  result = full_simplex(sf, print_it=print_it)
  table, x, status = result.table, result.x, result.status
  
  if status != 'Óptimo':
    return SimplexResult(None, None, status)
  
  while not all_integer(x):
    Y, y0 = table.sf.A, table.sf.b
    
    r = np.argmax(np.array([frac(x) for x in y0]))
    
    cut = np.array([-frac(x) for x in Y[r, :]])
    cut_b = -frac(y0[r])
    
    sf = add_restriction(sf, cut, cut_b)
    
    result = full_simplex(sf, print_it=print_it)
    table, x, status = result.table, result.x, result.status
    
    if status != 'Óptimo':
        return SimplexResult(None, None, status)
    
  return result
    
    
# c = np.array([1, 1])
# A = np.array([
#   [-2, 2],
#   [2, -5],
#   [5, 3]])
# b = np.array([-1, 0, 30])
# E = np.empty((0,2))
# d = np.empty(0)

# lp = LinearProblem(c, A, b, E, d)
# result = gomory(lp)

# print(result.x)
# print(result.table.z)
# print(result.status)
