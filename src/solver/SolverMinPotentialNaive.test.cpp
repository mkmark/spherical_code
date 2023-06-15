#include <src/solver/SolverMinPotentialNaive.hpp>


int main(int argc, char* argv[])
{
  /// setting
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  /// input
  /// n
  int n = 20;
  /// seed
  srand48(std::stoi("42"));
  /// dump_base_path
  std::string dump_base_path = "";

  /// run
  /// solve
  auto solver_min_potential = SolverMinPotentialNaive(n, dump_base_path);
  solver_min_potential.solve();

  /// output
  /// value
  std::cout<<solver_min_potential.value<<std::endl;
  /// step
  std::cout<<solver_min_potential.step<<std::endl;
  /// diff
  std::cout<<solver_min_potential.diff<<std::endl;
  return 0;
}
