#include <vector>
#include <iostream>
#include <random>
#include <float.h>
#include <chrono>
#include <cmath>

// #include <omp.h>

#include "src/solver/SolverBaseTorch.hpp"
#include "src/solver/Vector3.hpp"

#include <torch/torch.h>

template <typename T>
class SolverMinPotentialNaiveTorch : public SolverBaseTorch<T>{
public:
  T temperature;
  T cooling_factor;
  int cooling_step_count;
  // T cooling_delta;

  T d;

  SolverMinPotentialNaiveTorch(
    int n,
    std::string dump_base_path = "",
    torch::Tensor c_points_init = torch::Tensor(),
    T tol = 1e-15,
    T alpha_init = 0.1,
    int max_step = INT32_MAX
  ) : SolverBaseTorch<T>(
    n,
    dump_base_path,
    c_points_init,
    tol,
    alpha_init,
    max_step
  )
  {
    c_points_init = torch::rand({this->n, 3}, torch::device(at::kCUDA));

    this->alpha = 100.0/n/n;

    d = 2.199/sqrt(this->n)/20;

    temperature = 2.199/sqrt(this->n);
    cooling_factor = 0.9996;
    cooling_step_count = 10000;
    // cooling_delta = 0.1*temperature/cooling_step_count;
    this->min_step = cooling_step_count;
  }

  void gen_grads(){
    auto directions = this->c_points.unsqueeze(1).expand({-1, this->n, -1}, true) - this->c_points.unsqueeze(0).expand({-1, this->n, -1}, true);
    auto norms = torch::norm(directions, 2, 2).unsqueeze(2);
    this->grads = directions/norms/norms/norms.nan_to_num().sum(0);
    torch::Tensor tmp = this->grads.max();
    this->alpha = d/(tmp.item<T>());
    norms = 1/norms;
    tmp = norms.nan_to_num().sum();
    this->value = tmp.item<T>()/2;
  }

  void before_step(){
    if (this->step < cooling_step_count){
      this->c_points += temperature * (torch::rand({this->n, 3}, torch::device(at::kCUDA)) - 0.5);
      temperature *= cooling_factor;
    }
  }
};
