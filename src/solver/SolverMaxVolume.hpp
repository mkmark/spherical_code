#include <vector>
#include <iostream>
#include <random>
#include <float.h>
#include <chrono>
#include <cmath>

#include <omp.h>

#include "include/quickhull/QuickHull.cpp"

#include "src/solver/SolverBase.hpp"
#include "src/solver/Vector3.hpp"


template <typename T>
class SolverMaxVolume : public SolverBase<T>{
public:
  T temperature;
  T cooling_factor;
  int cooling_step_count;
  // T cooling_delta;

  SolverMaxVolume(
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

    temperature = 2.199/sqrt(this->n);
    cooling_factor = 0.9996;
    cooling_step_count = 10000;
    // cooling_delta = 0.1*temperature/cooling_step_count;
    this->min_step = cooling_step_count;
  }

  Vector3<T> direction;
  Vector3<T> grad;
  quickhull::QuickHull<T> qh;
  quickhull::ConvexHull<T> hull;
  std::vector<unsigned long> indexBuffer = std::vector<unsigned long>();

  Vector3<T> pV_px(Vector3<T> a, Vector3<T> b){
    return Vector3<T>(
      a.y*b.z-a.z*b.y,
      a.z*b.x-a.x*b.z,
      a.x*b.y-a.y*b.x
    );
  }

  void gen_grads(){
    for (auto& grad : this->grads){
      grad.x = 0;
      grad.y = 0;
      grad.z = 0;
    }

    if (this->step % 1000 == 0){
      hull = qh.getConvexHull(this->c_points, true, true, 1e-7);
      indexBuffer = hull.getIndexBuffer();
    }

    for (int i = 0; i < indexBuffer.size(); i+=3){
      auto ia = indexBuffer[i];
      auto ib = indexBuffer[i+1];
      auto ic = indexBuffer[i+2];
      auto a = this->c_points[ia];
      auto b = this->c_points[ib];
      auto c = this->c_points[ic];

      this->value -= (a.y*b.z*c.x + a.x*b.y*c.z + a.z*b.x*c.y - a.y*b.x*c.z - a.z*b.y*c.x - a.x*b.z*c.y)/6;

      this->grads[ia] += pV_px(b, c);
      this->grads[ib] += pV_px(c, a);
      this->grads[ic] += pV_px(a, b);
    }
  }

  bool is_validated(){
    this->value_prev = this->value;
    this->value = 0;

    for (auto& grad : this->grads){
      grad.x = 0;
      grad.y = 0;
      grad.z = 0;
    }

    hull = qh.getConvexHull(this->c_points, true, true, 1e-7);
    indexBuffer = hull.getIndexBuffer();

    for (int i = 0; i < indexBuffer.size(); i+=3){
      auto ia = indexBuffer[i];
      auto ib = indexBuffer[i+1];
      auto ic = indexBuffer[i+2];
      auto a = this->c_points[ia];
      auto b = this->c_points[ib];
      auto c = this->c_points[ic];

      this->value -= (a.y*b.z*c.x + a.x*b.y*c.z + a.z*b.x*c.y - a.y*b.x*c.z - a.z*b.y*c.x - a.x*b.z*c.y)/6;

      this->grads[ia] += pV_px(b, c);
      this->grads[ib] += pV_px(c, a);
      this->grads[ic] += pV_px(a, b);
    }

    for (int i = 0; i < this->n; ++i){
      this->c_points[i] -= this->alpha * this->grads[i];
      this->c_points[i].normalize();
    }

    ++this->step;
    this->diff = this->value_prev - this->value;
    return std::abs(this->diff) < this->tol;
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
