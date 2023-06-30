#include <vector>
#include <iostream>
#include <random>
#include <float.h>
#include <chrono>
#include <cmath>

// #include <omp.h>

#include "src/solver/SolverBase.hpp"
#include "src/solver/Vector3.hpp"


template <typename T>
class SolverMinPotentialNaive : public SolverBase<T>{
public:
  T temperature;
  T cooling_factor;
  int cooling_step_count;
  // T cooling_delta;

  T d;

  SolverMinPotentialNaive(
    int n,
    std::string dump_base_path = "",
    std::vector<Vector3<T>> c_points_init = std::vector<Vector3<T>>(),
    T tol = 1e-15,
    T alpha_init = 0.1,
    int max_step = INT32_MAX
  ) : SolverBase<T>(
    n,
    dump_base_path,
    c_points_init,
    tol,
    alpha_init,
    max_step
  )
  {
    for(int i=0; i<n; ++i) {
      this->c_points[i].x = drand48();
      this->c_points[i].y = drand48();
      this->c_points[i].z = drand48();
      this->c_points[i].normalize();
    }

    this->alpha = 100.0/n/n;

    d = 2.199/sqrt(this->n)/20;

    temperature = 2.199/sqrt(this->n);
    cooling_factor = 0.9996;
    cooling_step_count = 10000;
    // cooling_delta = 0.1*temperature/cooling_step_count;
    this->min_step = cooling_step_count;
  }

  void gen_grads(){
    T max_grad = 0;
    for (auto& grad : this->grads){
      grad.x = 0;
      grad.y = 0;
      grad.z = 0;
    }

    for (int i = 0; i < this->n; ++i){
      for (int j = i+1; j < this->n; ++j){
        auto direction = this->c_points[i] - this->c_points[j];

        auto norm2 = direction.getLengthSquared();
        auto norm = std::sqrt(norm2);
        auto grad = direction / (norm2 * norm);

        T v = 1/norm;
        this->value += v;
        max_grad = std::max(max_grad, v);

        this->grads[i] -= grad;
        this->grads[j] += grad;
      }
    }

    this->alpha = d/max_grad;
  }

  void before_step(){
    if (this->step < cooling_step_count){
      for(int i=0; i < this->n; i++) {
        this->c_points[i].x += temperature * (drand48() - 0.5);
        this->c_points[i].y += temperature * (drand48() - 0.5);
        this->c_points[i].z += temperature * (drand48() - 0.5);

        this->c_points[i].normalize();
      }

      temperature *= cooling_factor;
    }
  }
};
