import numpy as np

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