import numpy as np
from dataclasses import dataclass, field as dataclass_field

@dataclass
class LinearProblem:
  """
  problema de optimización lineal en la forma:
  min c^T x
  s.a. Ax <= b
       Ex = d
  """
  c: np.ndarray
  A: np.ndarray
  b: np.ndarray
  E: np.ndarray
  d: np.ndarray

  def __str__(self):
    return (f"LinearProblem:\n"
        f"c: {[str(x) for x in self.c]}\n"
        f"A: {[([str(x) for x in row]) for row in self.A]}\n"
        f"b: {[str(x) for x in self.b]}\n"
        f"E: {[[str(x) for x in row] for row in self.E]}\n"
        f"d: {[str(x) for x in self.d]}")
  
@dataclass
class StandardForm:
  """
  problema de optimización lineal en la forma estándar:
  min c^T x
  s.a. Ax = b
       x >= 0
  """
  c: np.ndarray
  A: np.ndarray
  b: np.ndarray
  xB: np.ndarray
  n: int # número de variables originales
  m: int # número de restricciones originales
  
  def __str__(self):
    return (f"StandardForm:\n"
            f"c: {[str(x) for x in self.c]}\n"
            f"A: {[[str(x) for x in row] for row in self.A]}\n"
            f"b: {[str(x) for x in self.b]}\n"
            f"xB: {[str(x) for x in self.xB]}\n"
            f"n: {self.n}\n"
            f"m: {self.m}")
  