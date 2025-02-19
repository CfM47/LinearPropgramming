import numpy as np
from src.opt_types import StandardForm

def add_restriction(
  sf: StandardForm, 
  new_row: np.ndarray, 
  new_b: float
):
    """
    Agrega una nueva restricción a la tabla simplex óptima
    """
    c, A, y0, xB = sf.c, sf.A, sf.b, sf.xB
    basic = [i for i, y in enumerate(new_row) if y != 0 and i in xB]

    # eliminar las variables basicas
    for i_b in basic:
      y_b = new_row[i_b]
      basic_row = np.where(xB == i_b)[0][0]
      new_row -= y_b * A[basic_row, :]
      new_b -= y_b * y0[basic_row]

    A = np.vstack([A, new_row])
    y0 = np.append(y0, new_b)

    m, n = A.shape
    new_xh = np.zeros((m, 1))
    new_xh[-1, 0] = 1
    A = np.hstack([A, new_xh])

    xB = np.append(xB, n)
  
    c = np.append(c, 0)
    
    new_sf = StandardForm(c, A, y0, xB, sf.n, sf.m + 1)
    
    return new_sf

# c = np.array([1, 3, 1, 0, 0])
# A = np.array([
#   [1, 2, -1, -1, 0], 
#   [0, 3, -1, -2, 1]])
# y0 = np.array([4, 12])
# xB = np.array([0, 4])
# new_row = np.array([1, 1, 1, 0, 0])
# new_b = 3

# sf = StandardForm(c, A, y0, xB, len(c) - len(xB), len(xB))

# sf = add_restriction(sf, new_row, new_b)

# print("A_updated:\n", sf.A)
# print("y0_updated:\n", sf.b)
# print("c_updated:\n", sf.c)
# print("xB_updated:\n", sf.xB)
  