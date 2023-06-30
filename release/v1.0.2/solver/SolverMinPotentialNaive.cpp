#include "SolverMinPotentialNaive.hpp"


int main(int argc, char* argv[])
{
  /// setting
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  /// input
  /// n
  int n = std::stoi(argv[1]);
  /// seed
  srand48(std::stoi(argv[2]));
  /// dump_base_path
  std::string dump_base_path = "";
  if (argc > 3) dump_base_path = argv[3];

  /// run
  /// solve
  // auto begin = std::chrono::steady_clock::now();
  auto solver_min_potential = SolverMinPotentialNaive<double>(n, dump_base_path);
  solver_min_potential.solve();
  // auto end = std::chrono::steady_clock::now();
  /// output
  /// value
  std::cout << solver_min_potential.value << std::endl;
  /// step
  std::cout << solver_min_potential.step << std::endl;
  /// diff
  std::cout << solver_min_potential.diff << std::endl;
  /// time
  // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
  
  return 0;
}
