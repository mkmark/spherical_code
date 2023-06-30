# %%
import numpy as np
from src.solver.solver_base_taichi import SolverBaseTaichi
import taichi as ti
ti.init(arch=ti.gpu, random_seed=42, default_ip=ti.i64)


@ti.data_oriented
class SolverMinPotentialTaichi(SolverBaseTaichi):
  def __init__(
    self,
    n: int=0,
    c_points_init: ti.MatrixField=None,
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
    n = self.c_points.shape[0]
    self.n = n
    self.alpha = 100/self.n/self.n
    self.temperature = 2.199/np.sqrt(self.n)
    self.cooling_factor = 0.9996
    self.cooling_step_count = 10000
    self.min_step = self.cooling_step_count

    self.d_tmp = 2.199/np.sqrt(self.n)/20

    ij_cnt = int(self.n*(self.n-1)/2)
    self.ij_cnt = ij_cnt
    
    # ijs = ti.Vector.field(2, ti.int32, shape=(self.ij_cnt))
    # @ti.kernel
    # def gen_ijs():
    #   for k in ti.ndrange(ij_cnt):
    #     i = int(ti.floor(n - 0.5 - ti.sqrt(n*n-n+0.25-2*k)))
    #     j = int(((i+3)/2 - n)*i) + 1 + k
    #     ijs[k] = [i, j]
    #   # k = 0
    #   # for i in range(self.n):
    #   #   for j in range(i+1, self.n):
    #   #     self.ijs[k] = [i,j]
    #   #     k += 1
    # gen_ijs()
    # self.ijs = ijs
    # print(self.ijs)

  @ti.kernel
  def jitter_c_points(self):
    for i in self.c_points:
      for j in ti.static(range(3)):
        #ti.static unrolls the inner loops
        self.c_points[i][j] += self.temperature * (ti.random() - 0.5)

  @ti.kernel
  def cal_grads(self) -> ti.f64:
    value: ti.f64 = 0

    ## high gpu usage, nan when n is large
    # n: ti.i64 = self.n
    # for k in ti.ndrange(self.ij_cnt):
    #   i = int(ti.floor(n - 0.5 - ti.sqrt(n*n-n+0.25-2*k)))
    #   j = int(((i+3)/2 -n)*i) + 1 + k
    #   # i, j = self.ijs[k]
    #   direction = self.c_points[i] - self.c_points[j]
    #   norm = direction.norm()
    #   grad = direction/norm/norm/norm
    #   value += 1/norm

    #   self.grads[i] -= grad                        
    #   self.grads[j] += grad

    # return value
  
    # n: ti.i64 = self.n
    # for i, j in ti.ndrange(self.n, self.n):
    #   if i == j:
    #     continue
    #   direction = self.c_points[i] - self.c_points[j]
    #   norm = direction.norm()
    #   grad = direction/norm/norm/norm
    #   value += 1/norm

    #   self.grads[j] += grad
    # return value/2
  
    ## high gpu usage
    for i in self.c_points:
      for j in range(i+1, n):
        direction = self.c_points[i] - self.c_points[j]
        norm = direction.norm()
        grad = direction/norm/norm/norm
        value += 1/norm

        self.grads[j] += grad
    return value
  

  def gen_grads(self) -> None:
    # simulated crystallization
    if (self.step < self.cooling_step_count):
      self.jitter_c_points()
      self.temperature *= self.cooling_factor
    self.alpha = self.d_tmp/ti.max(self.grads)[0]
    self.value = self.cal_grads()


# %% test
if __name__ == "__main__":
  from src.solver.solver_random_taichi import SolverRandomTaichi
  from src.solver.solver_chain import SolverChain
  from src.solver.solver_orientation import SolverOrientation

  # n = 1048576
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
      SolverRandomTaichi(),
      SolverMinPotentialTaichi(
        is_to_print_final=True,
        print_interval=1000,
        # is_to_visualize=True,
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
  
  time_start = datetime.datetime.now()
  solvers.solve()
  time_stop = datetime.datetime.now()
  print(time_stop - time_start)

# %%
if __name__ == "__main__":
  from scipy.spatial import ConvexHull
  import matplotlib.pyplot as plt
  import numpy as np

  c_points = solvers.solvers[-1].c_points.to_numpy()
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
