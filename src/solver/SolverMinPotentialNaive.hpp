#include <vector>
#include <iostream>
#include <random>
#include <float.h>
#include <chrono>
#include <cmath>

#include <src/solver/SolverBase.hpp>
#include <src/solver/util.hpp>

#include <omp.h>


class SolverMinPotentialNaive : public SolverBase{
public:
  // double temperature;
  // double cooling_factor;
  // int cooling_step_count;
  // double cooling_delta;

  SolverMinPotentialNaive(
    int n,
    std::string dump_base_path = "",
    std::vector<double> c_points_init = std::vector<double>(),
    double tol = 1e-15,
    double alpha_init = 0.1,
    int max_step = INT32_MAX
  ):
  SolverBase(
    n,
    dump_base_path,
    c_points_init,
    tol,
    alpha_init,
    max_step
  )
  {
    for(int i=0; i<n3; i++) {
      c_points[i] = drand48();
    }

    for (int i=0; i<n3; i+=3){
      auto cp_i = c_points.begin() + i;
      c_point_self_div(cp_i, c_point_l2norm(cp_i));
    }

    alpha = 100.0/n/n;

    // temperature = sqrt(8*sqrt(3)*M_PI/9/n);
    // cooling_factor = 0.9;
    // cooling_step_count = 40;
    // // cooling_delta = 0.1*temperature/cooling_step_count;
    // min_step = cooling_step_count;
  }

  std::vector<double> direction = std::vector<double>(3);
  std::vector<double> grad = std::vector<double>(3);

  void gen_grads(){
    std::fill(grads.begin(), grads.end(), 0);

    for (int i=0; i<n3; i+=3){
      auto cp_i = c_points.begin() + i;
      for (int j=i+3; j<n3; j+=3){
        c_point_sub(cp_i, c_points.begin() + j, direction.begin());
        auto norm2 = c_point_l2norm2(direction.begin());
        auto norm = std::sqrt(norm2);
        c_point_div(direction.begin(), norm2*norm, grad.begin());
        value += 1/norm;

        c_point_self_sub(grads.begin() + i, grad.begin());
        c_point_self_add(grads.begin() + j, grad.begin());
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
