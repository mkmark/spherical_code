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
  // T temperature;
  // T cooling_factor;
  // int cooling_step_count;
  // T cooling_delta;

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

    // temperature = sqrt(8*sqrt(3)*M_PI/9/n);
    // cooling_factor = 0.9;
    // cooling_step_count = 40;
    // // cooling_delta = 0.1*temperature/cooling_step_count;
    // min_step = cooling_step_count;
  }

  void gen_grads(){
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

        this->value += 1/norm;

        this->grads[i] -= grad;
        this->grads[j] += grad;
      }
    }
  }

  void before_step(){
    // if (step < cooling_step_count){
    //   for(int i=0; i<n3; i++) {
    //     c_points[i] += temperature * (drand48() - 0.5);
    //   }

    //   for (int i=0; i<n3; i+=3){
    //     auto cp_i = c_points.begin() + i;
    //     c_point_self_div(cp_i, c_point_l2norm(cp_i));
    //   }

    //   temperature *= cooling_factor;
    // }
  }
};
