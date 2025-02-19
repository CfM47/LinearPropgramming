import numpy as np
from src.dual_simplex import dual_simplex, is_dual_feasible
from src.simplex_twophase import twophase
from src.opt_types import StandardForm, SimplexResult, SimplexTable, LinearProblem
from fractions import Fraction
from src.utils import standarize, rationalize, make_integer

def full_simplex(
  sf: StandardForm,
  print_it: bool = False
) -> SimplexResult:
  
  if np.all(sf.b >= 0):
    return twophase(sf, print_it)
  
  c, A, b, xB = sf.c, sf.A, sf.b, sf.xB
  r = c - c[xB] @ A
  z = c[xB] @ b
  table = SimplexTable(sf, r, z)
  if is_dual_feasible(table):
    return dual_simplex(sf, print_it)
  
  return twophase(sf, print_it)