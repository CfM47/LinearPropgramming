import numpy as np
from src.pivot import pivot
from src.utils import print_table
from src.opt_types import StandardForm, SimplexTable, SimplexResult
from fractions import Fraction

def find_pivot_row(A: np.ndarray, q: int, y0: np.ndarray):
  """
  Encuentra la fila de pivote
  """
  p = -1
  min = np.inf
  for i in range(len(y0)):
    if A[i, q] <= 0:
      continue
    ratio = y0[i] / A[i, q]
    if ratio < min:
      min = ratio
      p = i
  return p

def simplex(
  sf: StandardForm,
  print_it: bool = False
) -> SimplexResult:
  """
  Algoritmo simplex, toma un problema de optimización lineal en la forma estandar:
  min c^T x
  s.a. Ax = b
       x >= 0
  """
  c, A, b, xB = sf.c, sf.A, sf.b, sf.xB
  r = c - c[xB] @ A
  z = c[xB] @ b
  
  if print_it:
    print_table(A, b, r, z)
  
  table = SimplexTable(sf, r, z)
  
  while np.any(table.r < 0): # criterio de optimalidad 
    r, z, A, y0, xB = table.r, table.z, table.sf.A, table.sf.b, table.sf.xB
    
    q = np.argmin(r) # criterio de entrada
    
    p = find_pivot_row(A, q, y0)
    if p < 0: 
      return SimplexResult(table, None, "Problema no acotado")
    
    
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