import numpy as np
from src.solver.util import visualize_c_points
from scipy.spatial import ConvexHull


class LoaderTxt():
  def __init__(
    self,
    n: int=0,
    is_to_print_final=False,
    is_to_visualize=True,
    load_base_path=None,
  ):
    self.n = n
    self.is_to_print_final = is_to_print_final
    self.is_to_visualize = is_to_visualize
    self.load_base_path = load_base_path

  def solve(self):
    self.c_points = np.loadtxt(f"{self.load_base_path}/{self.n}.txt").reshape(-1, 3)
    if self.is_to_print_final:
      self.print_res()

  def print_res(self):
    # clear_output()
    if self.is_to_print_final:
      print("solver: " + str(self.__class__))
    hull = ConvexHull(self.c_points)
    if self.is_to_visualize:
      visualize_c_points(self.c_points, hull)
