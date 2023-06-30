# %%
import numpy as np
from IPython.display import clear_output
from src.solver.util import visualize_c_points
from scipy.spatial import ConvexHull
import os
import pickle
import torch
import tqdm


class SolverBaseTorch():
  def __init__(
    self,
    n: int=0,
    c_points_init: torch.Tensor=None,
    tol=1e-16,
    alpha_init=1,
    print_interval=torch.inf,
    is_to_print_final=False,
    max_step=np.inf,
    dump_base_path=None,
    load_base_path=None,
    is_to_visualize=False,
  ) -> None:
    self.max_step = max_step
    self.tol = tol
    self.n = n
    self.alpha = alpha_init
    self.c_points = c_points_init
    self.dump_base_path = dump_base_path
    self.load_base_path = load_base_path
    self.is_to_visualize = is_to_visualize
    self.print_interval = print_interval
    self.is_to_print_final = is_to_print_final

    self.step = 0
    self.value = np.inf

    self.min_step = -1

  def gen_grads(self) -> None:
    raise NotImplementedError

  def step_next(self) -> None:
    self.value_prev = self.value
    self.value = 0

    self.gen_grads()

    self.c_points -= self.alpha * self.grads

    self.after_step()
    self.c_points /= torch.norm(self.c_points, dim=1, keepdim=True)
    
    # self.s_points = to_s_points(self.c_points)
    # self.s_points[0,1] = 0
    # self.s_points[0,2] = np.pi/2
    # self.s_points[:,0] = 1
    # self.c_points = to_c_points(self.s_points)

  
  def before_solve(self):
    pass

  def after_step(self):
    pass

  def solve(self):
    if self.load_base_path:
      self.load_c_points()
    
    # self.grads = torch.zeros(self.c_points.shape)

    self.before_solve()

    progress = tqdm.tqdm()
    while self.step < self.min_step or self.step < self.max_step:
      self.step_next()
      if self.print_interval != np.inf and self.step%self.print_interval == 0:
        self.print_res()
      self.step += 1
      progress.update(1)

      if self.step <= self.min_step:
        continue
      if self.step > self.max_step:
        break
      if abs(self.value_prev - self.value) < self.tol:
        break

    if self.is_to_print_final:
      self.print_res()

    if self.dump_base_path:
      self.dump_pickle()
    
  def print_res(self):
    # clear_output()
    print("solver: " + str(self.__class__))
    print("n: " + str(self.n))
    print("step: " + str(self.step))
    print("diff: " + str(abs(self.value_prev - self.value)))
    print("value: " + str(self.value))

    if self.is_to_visualize:
      c_points = self.c_points.cpu().numpy()
      hull = ConvexHull(c_points)
      visualize_c_points(c_points, hull)
    
  def get_pickle_path(
    self,
    base_path: str,
  ) -> str:
    return f"{base_path}/{self.n}.pickle"
  
  def dump_pickle(
    self,
  ) -> None:
    os.makedirs(self.dump_base_path, exist_ok=True)
    path = self.get_pickle_path(self.dump_base_path)

    self.c_points = self.c_points.cpu().numpy()
    self.grads = self.grads.cpu().numpy()

    with open(path, "wb") as f:
      pickle.dump(self, f)

  def load_c_points(
    self,
  ) -> None:
    path = self.get_pickle_path(self.load_base_path)
    with open(path, "rb") as f:
      solver = pickle.load(f)
    self.c_points = solver.c_points

# %%
