#pragma once

#include <cmath>
// #include <omp.h>
#include <vector>

inline void to_c_point(std::vector<double>::iterator s_point, std::vector<double>::iterator c_point){
  double r = *s_point;
  double theta = *(s_point+1);
  double phi = *(s_point+2);
  *c_point = r*cos(phi)*cos(theta);
  *(c_point+1) = r*cos(phi)*sin(theta);
  *(c_point+2) = r*sin(phi);
}

inline void to_s_point(std::vector<double>::iterator c_point, std::vector<double>::iterator s_point){
  double x = *c_point;
  double y = *(c_point+1);
  double z = *(c_point+2);
  double xy2 = x*x + y*y;
  /// r
  *s_point = sqrt(xy2 + z*z);
  /// theta
  *(s_point+1) = atan2(y,x);
  /// phi
  *(s_point+2) = atan2(z, sqrt(xy2));
}

inline void to_c_points(std::vector<double> & s_points, std::vector<double> & c_points){
  #pragma omp parallel for
  for (int i = 0; i < s_points.size(); i+=3) {
    to_c_point(s_points.begin() + i, c_points.begin() + i);
  }
}

inline void to_s_points(std::vector<double> & c_points, std::vector<double> & s_points){
  #pragma omp parallel for
  for (int i = 0; i < c_points.size(); i+=3) {
    to_s_point(c_points.begin() + i, s_points.begin() + i);
  }
}

inline void c_point_self_add(std::vector<double>::iterator a, std::vector<double>::iterator b) {
  *a += *b;
  *(a+1) += *(b+1);
  *(a+2) += *(b+2);
}

inline void c_point_self_sub(std::vector<double>::iterator a, std::vector<double>::iterator b) {
  *a -= *b;
  *(a+1) -= *(b+1);
  *(a+2) -= *(b+2);
}

inline void c_point_self_tms(std::vector<double>::iterator a, double s) {
  *a *= s;
  *(a+1) *= s;
  *(a+2) *= s;
}

inline void c_point_self_div(std::vector<double>::iterator a, double s) {
  *a /= s;
  *(a+1) /= s;
  *(a+2) /= s;
}

inline void c_point_add(std::vector<double>::iterator a, std::vector<double>::iterator b, std::vector<double>::iterator r) {
  *r = *a + *b;
  *(r+1) = *(a+1) + *(b+1);
  *(r+2) = *(a+2) + *(b+2);
}

inline void c_point_sub(std::vector<double>::iterator a, std::vector<double>::iterator b, std::vector<double>::iterator r) {
  *r = *a - *b;
  *(r+1) = *(a+1) - *(b+1);
  *(r+2) = *(a+2) - *(b+2);
}

inline void c_point_tms(std::vector<double>::iterator a, double s, std::vector<double>::iterator r) {
  *r = *a * s;
  *(r+1) = *(a+1) * s;
  *(r+2) = *(a+2) * s;
}

inline void c_point_div(std::vector<double>::iterator a, double s, std::vector<double>::iterator r) {
  *r = *a / s;
  *(r+1) = *(a+1) / s;
  *(r+2) = *(a+2) / s;
}

inline double c_point_l2norm2(std::vector<double>::iterator a) {
  return *a * *a + *(a+1) * *(a+1) + *(a+2) * *(a+2);
}

inline double c_point_l2norm(std::vector<double>::iterator a) {
  return std::sqrt(c_point_l2norm2(a));
}

inline void vec_self_add(std::vector<double> & a, std::vector<double> & b) {
  // #pragma omp parallel for
  for (int i = 0; i < a.size(); ++i) {
    a[i] += b[i];
  }
}

inline void vec_self_mul(std::vector<double> & a, double s) {
  // #pragma omp parallel for
  for (int i = 0; i < a.size(); ++i) {
    a[i] *= s;
  }
}
