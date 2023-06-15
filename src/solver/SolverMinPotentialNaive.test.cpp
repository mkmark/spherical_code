#include <src/solver/SolverMinPotentialNaive.hpp>


int main(int argc, char* argv[])
{
  /// setting
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  /// input
  /// n
  int n = 20;
  /// seed
  srand48(42);
  /// dump_base_path
  std::string dump_base_path = "";

  /// run
  /// solve
  auto begin = std::chrono::steady_clock::now();
  auto solver_min_potential = SolverMinPotentialNaive<double>(n, dump_base_path);
  solver_min_potential.solve();
  auto end = std::chrono::steady_clock::now();
  /// output
  /// value
  std::cout << solver_min_potential.value << std::endl;
  /// step
  std::cout << solver_min_potential.step << std::endl;
  /// diff
  std::cout << solver_min_potential.diff << std::endl;
  /// time
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  return 0;
}
