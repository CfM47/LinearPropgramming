import numpy as np
from src.opt_types import LinearProblem, StandardForm
from fractions import Fraction

def print_table(
  A: np.ndarray,
  y0: np.ndarray,
  r: np.ndarray,
  z: np.ndarray,
) -> None:
  """
  Imprime la tabla del algoritmo simplex
  """
  T = np.vstack([np.hstack([A, y0.reshape(-1, 1)]), np.hstack([r, z])])
  print(T)
  
def standarize(
  lp: LinearProblem,
) -> StandardForm:
  """
  Estandariza un problema de optimización lineal
  """
  c, A, b, E, d = lp.c, lp.A, lp.b, lp.E, lp.d
  # convertir las igualdades en desigualdades
  # Ex = d -> Ex <= d y -Ex <= -d
  E = np.vstack([E, -E])
  d = np.hstack([d, -d])
  
  A = np.vstack([A, E])
  b = np.hstack([b, d])
  
  # agregar variables de holgura
  m, n = A.shape
  c = np.hstack([c, np.array([Fraction(0)]*m)])
  frac_eye = np.array([[Fraction(1) if i == j else Fraction(0) for j in range(m)] for i in range(m)])
  A = np.hstack([A, np.eye(m, dtype=Fraction)])
  
  # definir variables basicas
  xB = np.arange(n, n+m)
  
  return StandardForm(c, A, b, xB, n, m)

def rationalize(
  lp: LinearProblem,
) -> LinearProblem:
  """
  Convierte los coeficientes de un problema de optimización lineal
  a fracciones
  """
  c, A, b, E, d = lp.c, lp.A, lp.b, lp.E, lp.d
  c = np.array([Fraction(x) for x in c])
  if A.shape[0] > 0:
    A = np.array([[Fraction(x) for x in row] for row in A])
  else:
    A = np.empty((0, A.shape[1]), dtype=Fraction)
  b = np.array([Fraction(x) for x in b])
  if E.shape[0] > 0:
    E = np.array([[Fraction(x) for x in row] for row in E])
  else:
    E = np.empty((0, E.shape[1]), dtype=Fraction)  
  d = np.array([Fraction(x) for x in d])
  
  return LinearProblem(c, A, b, E, d)

def get_lcd(
  vector: np.ndarray[Fraction],
) -> int:
  """
  Obtiene el mínimo común denominador de un vector de fracciones
  """
  denoms = [x.denominator for x in vector]
  return np.lcm.reduce(denoms)

def make_integer(
  lp: LinearProblem,
) -> LinearProblem:
  """
  Convierte las restricciones de un ppl a otras equivalentes con coeficientes enteros
  """
  c, A, b, E, d = lp.c, lp.A, lp.b, lp.E, lp.d
  # convertir las restricciones a enteros
  for i in range(len(b)):
    lcm = get_lcd(np.append(A[i], b[i]))
    A[i] *= lcm
    b[i] *= lcm
  for i in range(len(d)):
    lcm = get_lcd(np.append(E[i], d[i]))
    E[i] *= lcm
    d[i] *= lcm  
    
  return LinearProblem(c, A, b, E, d)

# c = np.array([1, 1])
# A = np.array([
#   [-1, 1], 
#   [-1, 2.5], 
#   [5, 3]
# ])
# b = np.array([0.5, 0, 30])
# E = np.empty((0, 2))
# d = np.empty(0)

# lp = LinearProblem(c, A, b, E, d)
# print('problem:\n', lp)
# lp = rationalize(lp)
# print('rationalized:\n', lp)
# lp = make_integer(lp)
# print('integerized:\n', lp)
# sf = standarize(lp)
# print('standard form:\n', sf)
