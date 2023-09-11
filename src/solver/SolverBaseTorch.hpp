#include <vector>
#include <iostream>
#include <string>
#include <float.h>
#include <iomanip>
#include <fstream>
#include <limits>

#include <src/solver/Vector3.hpp>

#include <torch/torch.h>

class NotImplemented : public std::logic_error
{
public:
  NotImplemented() : std::logic_error("Function not yet implemented") { };
};

template <typename T>
class SolverBaseTorch{
public:
  int min_step = -1;
  int max_step;
  T tol;
  int n;
  T alpha;
  torch::Tensor c_points;
  torch::Tensor grads;
  std::string dump_base_path;
  std::string load_base_path;

  int step = 0;
  T value = DBL_MAX;
  T value_prev;
  T diff;

  SolverBaseTorch(
    int n = 0,
    std::string dump_base_path = "",
    torch::Tensor c_points_init = torch::Tensor(),
    T tol = 1e-15,
    T alpha_init = 0.001,
    int max_step = INT32_MAX
  ):
    n(n),
    c_points(c_points_init),
    tol(tol),
    alpha(alpha_init),
    max_step(max_step),
    dump_base_path(dump_base_path)
  {
    c_points = torch::resize(c_points, {n, 3});
    grads = torch::resize(grads, {n, 3});
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

    c_points /= torch::norm(c_points, 2, 1, true);
  }

  virtual bool is_validated(){
    return 1;
  }

  void solve(){
    while (1){
      step_next();
      ++step;

      if (step <= min_step) continue;
      if (step > max_step) break;
      diff = value_prev - value;
      if (std::abs(diff) < tol && is_validated()) break;
    }

    if (dump_base_path != ""){
      // dump_txt();
    }
  }

  std::string get_txt_path(std::string base_path){
    return base_path+"/"+std::to_string(n)+".txt";
  }

  // void dump_txt(){
  //   auto path = get_txt_path(dump_base_path);
  //   std::ofstream outfile(path);
  //   if (outfile.is_open())
  //   {
  //     outfile << std::setprecision(std::numeric_limits<T>::max_digits10);

  //     for (auto& elem : c_points)
  //     {
  //       outfile << elem.x << "\n";
  //       outfile << elem.y << "\n";
  //       outfile << elem.z << "\n";
  //     }

  //     outfile.close();
  //   }
  // }
};
