import numpy as np
from fractions import Fraction
from src.simplex import simplex
from src.pivot import pivot

def simplex_twophase(
  c: np.ndarray, 
  A: np.ndarray, 
  b: np.ndarray,
  E: np.ndarray = np.empty((0, 0)),
  d: np.ndarray = np.empty(0),
  print_it: bool = False
) -> np.ndarray:
  """
  Algoritmo simplex, toma un problema de optimización lineal en la forma:
  min c^T x
  s.a. Ax <= b
       x >= 0
  """
  
  m, n = A.shape
  
  p, _ = E.shape
  
  # caso en el que se hace una sola fase
  if p == 0 and b.all() >= 0:
    # agregar variables de holgura
    A = np.hstack([A, np.eye(m)])
    c = np.hstack([c, np.zeros(m)])
    xB = np.arange(n, n + m)
    return simplex(c, A, b, xB, print_it)
  
  # caso en el que se hace dos fases
  
  # construir el problema auxiliar

  c_aux = np.zeros(n + m)
  # agregar variables de holgura
  A_aux = np.hstack([A, np.eye(m)])
  E_aux = np.hstack([E, np.zeros((p, m))])
  
  # hacer todos los b positivos
  negative_b = np.array([i for i in range(m) if b[i] < 0], dtype=int)
  A_aux[negative_b] *= -1
  b_aux = np.copy(b)
  b_aux[negative_b] *= -1
  
  # agregar variables artificiales
  artificial_count = negative_b.size + p
  A_aux = np.hstack([A_aux, np.zeros((m, artificial_count))])
  E_aux = np.hstack([E_aux, np.zeros((p, artificial_count))])
  # agregar variables artificiales a los b negativos
  for i in negative_b:
    A_aux[i, n + m + i] = 1
  # agregar variables artificiales a las restricciones de igualdad
  for i in range(p):
    E_aux[i, n + m + negative_b.size + i] = 1
  # agregar variables artificiales al vector c
  c_aux = np.hstack([c_aux, np.ones(artificial_count)])
  # determinar las variables básicas
  xB_aux = np.array([i for i in range(n, n + m + artificial_count) if i not in negative_b])
  
  P = np.vstack([A_aux, E_aux])
  b_aux = np.hstack([b_aux, d])
  
  # resolver el problema auxiliar
  _, z_aux, y0_aux, status = simplex(c_aux, P, b_aux, xB_aux)
  
  if status != "Óptimo" or z_aux > 0:
    return None, None, None, "Infactible"
  
  # eliminar variables artificiales
  for i in range(m + p):
    # sacar a las variables artificiales de la base
    if xB_aux[i] < n + m:
      continue
    i_nonzero = np.array([j for j in range(n + m) if P[i, j] != 0])
    if i_nonzero.size == 0:
      P = np.delete(P, i, axis=0)
      continue
    k = i_nonzero[0]
    pivot(P, y0_aux, k, xB_aux[i])
    xB_aux[i] = k
  # eliminar las columnas de las variables artificiales
  P = np.delete(P, np.arange(n + m, n + m + artificial_count), axis=1)
  
  A = P
  c = np.hstack([c, np.zeros(m)])
  b = y0_aux
  xB = xB_aux
  
  # resolver segunda fase
  result, z, y0, status = simplex(c, A, b, xB)
  
  return result, z, y0, status


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
