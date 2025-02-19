import numpy as np
from src.pivot import pivot
from src.utils import print_table
from src.opt_types import StandardForm, SimplexResult, SimplexTable
from fractions import Fraction

def find_pivot_col(
  A: np.ndarray,
  p: int,
  r: np.ndarray
) -> int:
  """
  Encuentra la columna de pivote
  """
  q = -1
  min = np.inf
  for j in range(len(r)):
    if A[p, j] >= 0:
      continue
    ratio = -r[j] / A[p, j]
    if ratio < min:
      min = ratio
      q = j
  return q

def is_dual_feasible(
  table: SimplexTable,
) -> bool:
  r = table.r
  return np.all(r >= 0)
  

def dual_simplex(
  sf: StandardForm,
  print_it: bool = False
) -> SimplexResult:
  """
  Algoritmo dual simplex, toma un problema de optimización lineal en la forma estandar:
  min c^T x
  s.a. Ax = b
       x >= 0
       
  Nota: se asume que el problema cumple dual factibilidad
  """
  c, A, b, xB = sf.c, sf.A, sf.b, sf.xB
  r = c - c[xB] @ A
  z = c[xB] @ b
  
  if print_it:
    print_table(A, b, r, z)
  
  table = SimplexTable(sf, r, z)
  
  while np.any(table.sf.b < 0): # criterio de optimalidad dual
    r, z, A, y0, xB = table.r, table.z, table.sf.A, table.sf.b, table.sf.xB
    
    p = np.argmin(y0) # criterio de salida
    
    q = find_pivot_col(A, p, r)
    if q < 0:
      return SimplexResult(None, None, "Infactible")
    
    table.sf = pivot(table.sf, q, p)
    A, y0, xB = table.sf.A, table.sf.b, table.sf.xB
    r = c - c[xB] @ A
    z = c[xB] @ y0
    
    table = SimplexTable(table.sf, r, z)
    
    if print_it:
      print_table(A, y0, r, z)
  
  n = len(c)
  status = "Óptimo"
  x = np.array([Fraction(0)] * n)
  x[xB] = y0
  return SimplexResult(table, x, status)

# c = np.array([3, 2, 1, 0, 0])
# A = np.array([
#   [-1, -1, 1, 1, 0],
#   [2, 1, -2, 0, 1]
# ])
# b = np.array([-5, -4])
# xB = np.array([3, 4])

# r, z, y0, status = dual_simplex(c, A, b, xB)

# print("Resultado:", r)
# print("Valor óptimo:", z)
# print("Variables básicas:", y0)
# print("Estado:", status)