import numpy as np

def pivot(
  A: np.ndarray, 
  y0: np.ndarray, 
  q: int, 
  p: int
) -> np.ndarray:
  """
  Realiza la operación de pivoteo en el algoritmo simplex
  """
  m, _ = A.shape
  y_pq = float(A[p, q])
  A[p] = A[p] / y_pq
  y0[p] = y0[p] / y_pq
  
  for i in range(m):
    if i == p:
      continue
    y_iq = float(A[i, q])
    A[i] = A[i] - y_iq * A[p]
    y0[i] = y0[i] - y_iq * y0[p]