import numpy as np
from src.pivot import pivot
from src.utils import print_table

def simplex(
  c: np.ndarray, 
  A: np.ndarray, 
  b: np.ndarray,
  xB: np.ndarray,
  print_it: bool = False
) -> np.ndarray:
  """
  Algoritmo simplex, toma un problema de optimización lineal en la forma estandar:
  min c^T x
  s.a. Ax = b
       x >= 0
  """
  m, n = A.shape
  
  r = c - c[xB] @ A
  y0 = b
  z = c[xB] @ b
  
  if print_it:
    print_table(A, y0, r, z)
  
  while np.any(r < 0): # criterio de optimalidad 
    q = np.argmin(r) # criterio de entrada
    if(np.all(A[:, q] <= 0)):
      status = "Problema no acotado"
      return None, None, None, status
    
    ratios = np.where(A[:, q] > 0, y0 / A[:, q].astype(float), np.inf)
    p = np.argmin(ratios) # criterio de salida
    
    pivot(A, y0, q, p)
    
    xB[p] = q      
    r = c - c[xB] @ A
    z = c[xB] @ y0
    
    if print_it:
      print_table(A, y0, r, z)
  
  status = "Óptimo"
  result = np.zeros(n)
  result[xB] = y0
  return result, z, y0, status