import numpy as np
from src.opt_types import StandardForm

def pivot(
  sf: StandardForm,
  q: int, 
  p: int
) -> StandardForm:
  """
  Realiza la operación de pivoteo en el algoritmo simplex
  """
  A, y0 = sf.A, sf.b
  
  m, _ = A.shape
  y_pq = A[p, q]
  A[p] = A[p] / y_pq
  y0[p] = y0[p] / y_pq
  
  for i in range(m):
    if i == p:
      continue
    y_iq = A[i, q]
    A[i] = A[i] - y_iq * A[p]
    y0[i] = y0[i] - y_iq * y0[p]
  
  xB = sf.xB
  xB[p] = q
  return StandardForm(sf.c, A, y0, xB, sf.n, sf.m)