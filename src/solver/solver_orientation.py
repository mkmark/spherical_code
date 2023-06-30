# %%
import numpy as np
from scipy.spatial import ConvexHull
from src.solver.solver_base import SolverBase
import torch

class SolverOrientation(SolverBase):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def solve(self) -> None:
    if type(self.c_points) == torch.Tensor:
      if self.c_points.is_cuda:
        self.c_points = self.c_points.cpu()
      self.c_points = self.c_points.numpy()
    
    hull = ConvexHull(self.c_points)
    self.n = len(self.c_points)
    self.step = 0
    self.value_prev = 0
    self.value = 0

    # get id of point with minimal neighbours, as p0
    neighbour_counts = np.zeros(self.n, dtype=int)
    for simplex in hull.simplices:
      for pid in simplex:
        neighbour_counts[pid] += 1
    p0id = np.argmin(neighbour_counts)

    # get neighbours of p0, as p1s
    p1ids = set()
    for simplex in hull.simplices:
      if p0id in simplex:
        for pid in simplex:
          if pid != p0id:
            p1ids.add(pid)

    # get distance of p0 and p1s
    # select point of minimal distance as p1
    p1ids = list(p1ids)
    distances = np.zeros(len(p1ids))
    for i, p1id in enumerate(p1ids):
      distances[i] = np.linalg.norm(self.c_points[p1id] - self.c_points[p0id])
    p1id = p1ids[np.argmin(distances)]

    # orient the model such that p0 is set at (0, 0, 1)
    # and p1 at (1, ?, ?)
    c_p0 = self.c_points[p0id]
    c_p1 = self.c_points[p1id]

    # oz = c_p0
    # ox = np.cross(oz, c_p1)
    # oy = np.cross(oz, ox)
    # derive the orientation matrix of two 3d orthogonal coordinate systems in python

    # Define the orthonormal basis vectors for the two coordinate systems
    i1 = np.array([1, 0, 0])
    j1 = np.array([0, 1, 0])
    k1 = np.array([0, 0, 1])

    k2 = c_p0
    i2 = np.cross(k2, c_p1)
    j2 = np.cross(k2, i2)

    # Combine the basis vectors of coordinate system 2 into a matrix
    M = np.column_stack((i2, j2, k2))

    # Calculate the dot products of each pair of basis vectors between the two coordinate systems
    dot_i = np.dot(i1, M)
    dot_j = np.dot(j1, M)
    dot_k = np.dot(k1, M)

    orientation_matrix = np.column_stack((dot_i, dot_j, dot_k))
    
    self.c_points = (orientation_matrix@self.c_points.T).T
    
    if self.is_to_print_final:
      self.print_res()

    if self.dump_base_path:
      self.dump_pickle()


# %% test

if __name__ == "__main__":
  from src.solver.solver_random import SolverRandom
  from src.solver.solver_chain import SolverChain
  from src.solver.solver_min_potential_torch import SolverMinPotentialTorch

  n = 12


  solver_random = SolverRandom(
    n,
  )
  solver_random.solve()
  c_points_random = solver_random.c_points
  
  
  solver_max_potential = SolverMinPotentialTorch(
    n,
    c_points_random,
    print_interval=1000,
    is_to_print_final=True,
    max_step=np.inf,
  )
  solver_max_potential.solve()
  c_points_max_potential = solver_max_potential.c_points


  solver_orientation = SolverOrientation(
    c_points_init = c_points_max_potential,
    is_to_print_final = True,
  )
  solver_orientation.solve()



# %%
