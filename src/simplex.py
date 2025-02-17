import numpy as np
from src.pivot import pivot
from src.utils import print_table


def simplex(
  c: np.ndarray, 
  A: np.ndarray, 
  b: np.ndarray,
  xB: np.ndarray
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
  
  xR = np.setdiff1d(np.arange(n), xB) # indices de variables no basicas  
  
  print_table(A, y0, r, z)
  
  while np.any(r[xR] < 0): # criterio de optimalidad 
    q = np.argmin(r[xR]) # criterio de entrada
    if(np.all(A[:, q] <= 0)):
      status = "Problema no acotado"
      return None, None, status
    
    ratios = np.where(A[:, q] > 0, y0 / A[:, q].astype(float), np.inf)
    p = np.argmin(ratios) # criterio de salida
    
    pivot(A, y0, q, p)
    
    xB[p], xR[q] = xR[q], xB[p]
    r = c - c[xB] @ A
    z = c[xB] @ y0
    
    print_table(A, y0, r, z)
  
  status = "Óptimo"
  return r, z, status
    

c = np.array([2, 1, 3, 2, 1], dtype=float)
A = np.array([
  [1, 1, 1, 1, 0], 
  [-1, 1, 2, 0, 1]
  ], dtype=float)
b = np.array([9, 3], dtype=float)
xB = np.array([3, 4], dtype=int)

simplex(c, A, b, xB)