import numpy as np
from src.pivot import pivot
from src.utils import print_table

def is_dual_feasible(
  c: np.ndarray, 
  A: np.ndarray, 
  xB: np.ndarray
) -> bool:
  r = c - c[xB] @ A
  return np.all(r >= 0)
  

def dual_simplex(
  c: np.ndarray, 
  A: np.ndarray, 
  b: np.ndarray,
  xB: np.ndarray,
  print_it: bool = False
) -> np.ndarray:
  """
  Algoritmo dual simplex, toma un problema de optimización lineal en la forma estandar:
  min c^T x
  s.a. Ax = b
       x >= 0
       
  Nota: se asume que el problema cumple dual factibilidad
  """
  m, n = A.shape
  
  r = c - c[xB] @ A
  y0 = b
  z = c[xB] @ b
  
  if print_it:
    print_table(A, y0, r, z)
  
  while np.any(y0 < 0): # criterio de optimalidad dual
    p = np.argmin(y0) # criterio de salida
    if(np.all(A[p, :] >= 0)):
      status = "Infactible"
      return None, None, None, status
    
    ratios = np.where(A[p, :] < 0, -r / A[p, :].astype(float), np.inf)
    q = np.argmin(ratios) # criterio de entrada
    
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

c = np.array([3, 2, 1, 0, 0])
A = np.array([
  [-1, -1, 1, 1, 0],
  [2, 1, -2, 0, 1]
])
b = np.array([-5, -4])
xB = np.array([3, 4])

r, z, y0, status = dual_simplex(c, A, b, xB)

print("Resultado:", r)
print("Valor óptimo:", z)
print("Variables básicas:", y0)
print("Estado:", status)