# %%
import numpy as np
from src.solver.solver_base_torch import SolverBaseTorch
import torch


class SolverMinPotentialTorch(SolverBaseTorch):
  def __init__(
    self,
    n: int=0,
    c_points_init: torch.Tensor=None,
    tol=1e-15,
    alpha_init=0.1,
    print_interval=np.inf,
    is_to_print_final=False,
    max_step=np.inf,
    dump_base_path=None,
    load_base_path=None,
    is_to_visualize=False,
  ) -> None:
    super().__init__(
      n,
      c_points_init,
      tol,
      alpha_init,
      print_interval,
      is_to_print_final,
      max_step,
      dump_base_path,
      load_base_path,
      is_to_visualize,
    )
  
  def before_solve(self) -> None:
    self.n = len(self.c_points)
    self.alpha = 100/self.n/self.n
    self.temperature = 2.199/np.sqrt(self.n)
    self.cooling_factor = 0.99999
    self.cooling_step_count = 800000
    self.min_step = self.cooling_step_count

    self.d_tmp = 2.199/np.sqrt(self.n)/20

  def gen_grads(self) -> None:
    ## simulated crystallization
    if (self.step < self.cooling_step_count):
      self.c_points += self.temperature * (torch.rand(*self.c_points.shape, device=self.c_points.device, dtype=self.c_points.dtype) - 0.5)
      self.temperature *= self.cooling_factor

    directions = self.c_points.unsqueeze(1).expand(-1, self.n, -1) - self.c_points.unsqueeze(0).expand(self.n, -1, -1)
    norms = torch.norm(directions, dim=2).unsqueeze(2)
    self.grads = torch.nan_to_num(directions/norms/norms/norms).sum(dim=0)
    self.alpha = self.d_tmp/torch.max(self.grads)

    ## potential = 1/norm
    norms = 1/norms
    norms[norms == float("Inf")] = 0
    self.value = torch.nan_to_num(norms).sum().cpu().numpy()/2


# %% test
if __name__ == "__main__":
  from src.solver.solver_random_torch import SolverRandomTorch
  from src.solver.solver_chain import SolverChain
  from src.solver.solver_orientation import SolverOrientation

  n = 4096

  import datetime
  time = datetime.datetime.now().isoformat()


  # solver_random = SolverRandom(
  #   n,
  # )
  # solver_random.solve()
  # c_points_random = solver_random.c_points
  
  
  # solver_max_potential = SolverMinPotential(
  #   n,
  #   c_points_random,
  #   print_interval=1000,
  #   is_to_print_final=True,
  #   max_step=np.inf,
  # )
  # solver_max_potential.solve()



  solvers = SolverChain(
    n = n,
    solvers = [
      SolverRandomTorch(
        device=torch.device("cuda:0"),
        dtype=torch.double
      ),
      SolverMinPotentialTorch(
        is_to_print_final=True,
        print_interval=1000,
        # load_base_path="out/2023-05-12T16:37:05.659378/topology",
        # dump_base_path=f"out/{time}/min_potential"
      ),
      # SolverOrientation(
      #   is_to_print_final=True,
      #   is_to_visualize=True,
      #   # dump_base_path=f"out/{time}/orientation"
      # )
    ],
  )
  
  solvers.solve()

# %%
if __name__ == "__main__":
  from scipy.spatial import ConvexHull
  import matplotlib.pyplot as plt
  import numpy as np

  c_points = solvers.solvers[-1].c_points.cpu().numpy()
  hull = ConvexHull(c_points)

  plt.figure(
    figsize=(10,10),
    dpi=1200
  )
  ax = plt.axes(projection='3d')
  if not hull:
    hull = ConvexHull(c_points)
  for simplex in hull.simplices:
    simplex = np.append(simplex, simplex[0])
    ax.axis('equal')
    ax.axis('off')
    ax.plot(c_points[simplex, 0], c_points[simplex, 1], c_points[simplex, 2], 'k-', linewidth=0.5);
  # write_to_txt(c_pointoints)
  plt.show()

# %%
if __name__ == "__main__":
  c_points = solvers.solvers[-1].c_points.cpu().numpy()
  with open(f"out/manual/min_potential/txt_dump/{n}.txt", "w") as f:
    for c_point in c_points:
      for x in c_point:
        f.write(str(x)+'\n')

# %%
