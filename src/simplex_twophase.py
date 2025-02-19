import numpy as np
from fractions import Fraction
from src.simplex import simplex
from src.pivot import pivot
from src.opt_types import SimplexTable, SimplexResult, StandardForm, LinearProblem
from src.utils import rationalize, standarize

# Implementación del simplex dos fases del Introduction to algorithms

def build_auxiliar_sf(sf: StandardForm) -> StandardForm:
  """
  Construye el problema auxiliar de un problema de optimizacion lineal en forma estandar
  """
  A, b, xB = sf.A, sf.b, sf.xB
  m, n = A.shape
  c = np.hstack([np.zeros(n), 1])
  c = np.array([Fraction(x) for x in c])
  
  # Añadir una nueva variable con índice -1 a todas las restricciones de A
  A = np.hstack([A, [[Fraction(x) for x in row ] for row in -np.ones((m, 1))]])
  k = np.argmin(b)
  sf = StandardForm(c, A, b, xB, sf.n, sf.m)
  sf = pivot(sf, n, k)
  
  return sf
  
def get_original_sf(sf: StandardForm, c: StandardForm) -> StandardForm:
  """
  Devuelve un problema auxiliar a su forma original
  """
  xB, A = sf.xB, sf.A
  m, n = A.shape
  
  aux = n-1
  p = -1
  for i in range(m):
    if xB[i] == aux:
      p = i
      break

  if p == -1:
    A = np.delete(A, n-1, axis=1)
    return StandardForm(c, A, sf.b, xB, n-1, m)
  
  q = -1
  for i in range(n):
    if A[p, i] != 0:
      q = i
      break    
  if q == -1:
    A = np.delete(A, p, axis=0)
    A = np.delete(A, n-1, axis=1)
    xB = np.delete(xB, p)
    return StandardForm(c, A, sf.b, n-1, m-1)
  sf = pivot(sf, q, p)
  A = sf.A
  A = np.delete(A, n-1, axis=1)
  return StandardForm(c, A, sf.b, sf.xB, n-1, m)

def simplex_twophase(
  lp: LinearProblem,
  print_it: bool = False
) -> SimplexResult:
  """
  Algoritmo simplex, toma un problema de optimización lineal en la forma:
  min c^T x
  s.a. Ax <= b
       x >= 0
  """
  lp = rationalize(lp)
  sf = standarize(lp)  
    
  # caso en el que se hace una sola fase
  if np.all(sf.b >= 0):
    return simplex(sf, print_it)
  
  # caso en el que se hace dos fases
  sf_aux = build_auxiliar_sf(sf)
  result = simplex(sf_aux)
  
  if result.status != "Óptimo":
    return SimplexResult(None, None, "Infactible")
  
  if result.table is not None and result.table.z != 0:
    return SimplexResult(None, None, "Infactible")
  
  sf = get_original_sf(sf_aux, sf.c)
  # resolver segunda fase
  
  return simplex(sf)


# c = np.array([-3, -1, -2], dtype=Fraction)
# A = np.array([
#   [1, 1, 3], 
#   [2, 2, 5],
#   [4, 1, 2]
#   ], dtype=Fraction)
# b = np.array([30, 24, 36], dtype=Fraction)

# r, z, y0, status = simplex_twophase(c, A, b) 

# print(r, z, status) #[8, 4, 0, 18, 0, 0] -28 Óptimo

# c = np.array([4, 1, 1], dtype=float)
# A = np.empty((0, 3), dtype=float)
# b = np.empty(0, dtype=float)
# E = np.array([
#   [2, 1, 2],
#   [3, 3, 1]
#   ], dtype=float)
# d = np.array([4, 3], dtype=float)

# r, z, y0, status = simplex_twophase(c, A, b, E, d)
# print(r, z, status) #[0, 2/5, 9/5] 11/5 Óptimo
