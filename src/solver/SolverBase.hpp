#include <vector>
#include <iostream>
#include <string>
#include <float.h>
#include <iomanip>
#include <fstream>
#include <limits>

#include <src/solver/util.hpp>

class NotImplemented : public std::logic_error
{
public:
  NotImplemented() : std::logic_error("Function not yet implemented") { };
};

class SolverBase{
public:
  int min_step = -1;
  int max_step;
  double tol;
  int n;
  int n3;
  double alpha;
  std::vector<double> c_points;
  std::vector<double> s_points;
  std::vector<double> grads;
  std::string dump_base_path;
  std::string load_base_path;

  int step = 0;
  double value = DBL_MAX;
  double value_prev;
  double diff;

  SolverBase(
    int n = 0,
    std::string dump_base_path = "",
    std::vector<double> c_points_init = std::vector<double>(),
    double tol = 1e-15,
    double alpha_init = 0.001,
    int max_step = INT32_MAX
  ):
    n(n),
    c_points(c_points_init),
    tol(tol),
    alpha(alpha_init),
    max_step(max_step),
    dump_base_path(dump_base_path)
  {
    n3 = n*3;
    c_points.resize(n3);
    s_points.resize(n3);
    grads.resize(n3);
    value = DBL_MAX;
  }

  virtual void gen_grads(){
    throw NotImplemented();
  }

  virtual void before_step(){}

  void step_next(){
    value_prev = value;
    value = 0;

    before_step();

    gen_grads();

    for (int i = 0; i<n3; ++i){
      c_points[i] -= alpha * grads[i];
    }


    // to_s_points(c_points, s_points);
    // s_points[1] = 0;
    // s_points[2] = M_PI/2;
    // // #pragma omp parallel for
    // for (int i=0; i<n3; i+=3){
    //   s_points[i] = 1;
    // }
    // to_c_points(s_points, c_points);


    for (int i=0; i<n3; i+=3){
      auto cp_i = c_points.begin() + i;
      c_point_self_div(cp_i, c_point_l2norm(cp_i));
    }
  }

  void solve(){
    while (1){
      step_next();
      step += 1;

      if (step <= min_step) continue;
      if (step > max_step) break;
      diff = value_prev - value;
      if (std::abs(diff) < tol) break;
    }

    if (dump_base_path != ""){
      dump_txt();
    }
  }

  std::string get_txt_path(std::string base_path){
    return base_path+"/"+std::to_string(n)+".txt";
  }

  void dump_txt(){
    auto path = get_txt_path(dump_base_path);
    std::ofstream outfile(path);
    if (outfile.is_open())
    {
      outfile << std::setprecision(std::numeric_limits<double>::max_digits10);

      for (auto& elem : c_points)
      {
        outfile << elem << "\n";
      }

      outfile.close();
    }
  }
};
