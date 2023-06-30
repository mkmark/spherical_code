from src.solver.solver_base import SolverBase


class SolverChain():
  def __init__(
    self,
    n: int,
    solvers: list[SolverBase],
  ) -> None:
    self.solvers = solvers
    for solver in solvers:
      solver.n = n
  
  def solve(self):
    for i, solver in enumerate(self.solvers):
      if i > 0:
        solver.c_points = self.c_points
      solver.solve()
      self.c_points = solver.c_points
