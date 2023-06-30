import random
import numpy as np
from numpy import pi
from src.solver.util import to_c_points, visualize_c_points
from IPython.display import clear_output
from scipy.spatial import ConvexHull

random.seed(42)


class SolverRandom():
  def __init__(
    self,
    n: int=0,
    is_to_print_final=False,
  ):
    self.n = n
    self.is_to_print_final = is_to_print_final

  def solve(self):
    self.s_points = []
    self.s_points += [[1, 0, pi/2]]
    for i in range(1, self.n):
      theta = random.random() * 2 * pi
      phi = random.random() * pi - pi/2
      sp = [1, theta, phi]
      self.s_points += [sp]
        
    self.s_points = np.array(self.s_points)
    self.c_points = to_c_points(self.s_points)
    if self.is_to_print_final:
      self.print_res()

  def print_res(self):
    # clear_output()
    print("solver: " + str(self.__class__))
    hull = ConvexHull(self.c_points)
    visualize_c_points(self.c_points, hull)
