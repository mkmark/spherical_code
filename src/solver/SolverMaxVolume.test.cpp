#include <src/solver/SolverMaxVolume.hpp>


int main(int argc, char* argv[])
{
  /// setting
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  /// input
  /// n
  int n = 100;
  /// seed
  srand48(42);
  /// dump_base_path
  std::string dump_base_path = "";

  /// run
  /// solve
  auto begin = std::chrono::steady_clock::now();
  auto solver_max_volume = SolverMaxVolume<double>(n, dump_base_path);
  solver_max_volume.solve();
  auto end = std::chrono::steady_clock::now();
  /// output
  /// value
  std::cout << solver_max_volume.value << std::endl;
  /// step
  std::cout << solver_max_volume.step << std::endl;
  /// diff
  std::cout << solver_max_volume.diff << std::endl;
  /// time
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  return 0;
}
