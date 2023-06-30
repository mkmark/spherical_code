# %%
import numpy as np
from src.solver.solver_base import SolverBase


class SolverMinPotential(SolverBase):
  def __init__(
    self,
    n: int=0,
    c_points_init: np.ndarray=None,
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
    self.alpha = 0.1/self.n
    self.temperature = np.sqrt(8*np.sqrt(3)*np.pi/9/self.n)

  def after_step(self) -> None:
    ## simulated annealing
    self.c_points += self.temperature * (np.random.rand(*self.c_points.shape)-0.5)
    self.temperature *= 0.9999
    pass

  def gen_grads(self) -> None:
    self.grads.fill(0)

    for i, cp_i in enumerate(self.c_points):
      for j in range(i+1, self.n):
        direction = cp_i - self.c_points[j]
        norm = np.linalg.norm(direction, 2)
        grad = direction/norm/norm/norm
        self.value += 1/norm

        self.grads[i] -= grad                        
        self.grads[j] += grad


# %% test
if __name__ == "__main__":
  from src.solver.solver_random import SolverRandom
  from src.solver.solver_chain import SolverChain
  from src.solver.solver_orientation import SolverOrientation

  n = 128

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
      SolverRandom(),
      SolverMinPotential(
        is_to_print_final=True,
        print_interval=10,
        # load_base_path="out/2023-05-12T16:37:05.659378/topology",
        # dump_base_path=f"out/{time}/min_potential"
      ),
      SolverOrientation(
        is_to_print_final=True,
        # dump_base_path=f"out/{time}/orientation"
      )
    ],
  )
  
  solvers.solve()

# %%

# %%
