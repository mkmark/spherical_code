# %%

import random
import numpy as np
from numpy import pi
from src.solver.util import to_c_points, visualize_c_points
from IPython.display import clear_output
from scipy.spatial import ConvexHull

import taichi as ti
ti.init(arch=ti.gpu, random_seed=42)



class SolverRandomTaichi():
  def __init__(
    self,
    n: int=0,
    is_to_print_final=False,
  ):
    self.n = n
    self.is_to_print_final = is_to_print_final

  def solve(self):
    self.c_points: ti.MatrixField = ti.Vector.field(3, dtype=float, shape=(self.n,))
    
    @ti.kernel
    def fill_c_points():
      for i in self.c_points:
        for j in ti.static(range(3)):
          #ti.static unrolls the inner loops
          self.c_points[i][j] = ti.random() - 0.5
    fill_c_points()

    @ti.kernel
    def norm_c_points():
      for i in self.c_points:
        self.c_points[i] /= self.c_points[i].norm()
    norm_c_points()

    if self.is_to_print_final:
      self.print_res()

  def print_res(self):
    # clear_output()
    print("solver: " + str(self.__class__))
    c_points = self.c_points.to_numpy()
    hull = ConvexHull(c_points)
    visualize_c_points(c_points, hull)


# %% test
if __name__ == "__main__":
  solver = SolverRandomTaichi(
    5,
    is_to_print_final=True,
  )
  solver.solve()
  solver.c_points
  
# %%
