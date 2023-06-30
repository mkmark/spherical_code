# %%

import random
import numpy as np
from numpy import pi
from src.solver.util import to_c_points, visualize_c_points
from IPython.display import clear_output
from scipy.spatial import ConvexHull
import torch

random.seed(42)
torch.manual_seed(42)


class SolverRandomTorch():
  def __init__(
    self,
    n: int=0,
    is_to_print_final=False,
    device=torch.device("cuda"),
    dtype=torch.double,
  ):
    self.n = n
    self.is_to_print_final = is_to_print_final
    self.device = device
    self.dtype=dtype

  def solve(self):
    self.c_points = torch.rand(self.n, 3, dtype=self.dtype).to(self.device) - 0.5
    self.c_points /= torch.norm(self.c_points, dim=1, keepdim=True)
    if self.is_to_print_final:
      self.print_res()

  def print_res(self):
    # clear_output()
    print("solver: " + str(self.__class__))
    c_points = self.c_points.cpu().numpy()
    hull = ConvexHull(c_points)
    visualize_c_points(c_points, hull)


# %% test
if __name__ == "__main__":
  solver = SolverRandomTorch(
    5,
    is_to_print_final=True,
  )
  solver.solve()
  solver.c_points
  
# %%
