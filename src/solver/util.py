from enum import Enum
import numpy as np
import numexpr as ne
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

ceval = ne.evaluate
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def to_c_point(s_point: np.ndarray) -> np.ndarray:
  """convert point of spherical coordinate system to cartesian coordinate"""
  r = s_point[0]
  theta = s_point[1]
  phi = s_point[2]
  x = r*cos(phi)*cos(theta)
  y = r*cos(phi)*sin(theta)
  z = r*sin(phi)
  return([x, y, z])

def to_c_points(s_points: np.ndarray) -> np.ndarray:
  """convert points of spherical coordinate system to cartesian coordinate"""
  c_points = []
  for s_point in s_points:
    c_points += [to_c_point(s_point)]
  return np.array(c_points)

def to_s_point(c_point: np.ndarray) -> np.ndarray:
  """convert point of cartesian coordinate system to spherical coordinate"""
  x = c_point[0]
  y = c_point[1]
  z = c_point[2]
  theta = ceval('arctan2(y,x)')
  xy2 = ceval('x*x + y*y')
  phi = ceval('arctan2(z, sqrt(xy2))')
  r = ceval('sqrt(xy2 + z*z)')
  return([r, theta, phi])

def to_s_points(c_points: np.ndarray) -> np.ndarray:
  """convert points of cartesian coordinate system to spherical coordinate"""
  s_points = []
  for c_point in c_points:
    s_points += [to_s_point(c_point)]
  return np.array(s_points)

def get_volume(c_points: np.ndarray, hull: ConvexHull) -> float:
  v = 0
  for simplex in hull.simplices:
    a = c_points[simplex[0], :]
    b = c_points[simplex[1], :]
    c = c_points[simplex[2], :]
    v += abs(a[1]*b[2]*c[0] + a[0]*b[1]*c[2] + a[2]*b[0]*c[1] - a[1]*b[0]*c[2] - a[2]*b[1]*c[0] - a[0]*b[2]*c[1])
  return v/6  

# %matplotlib notebook
def visualize_c_points(
  c_points: np.ndarray,
  hull: ConvexHull = None,
):
  plt.figure(figsize=(10,6))
  ax = plt.axes(projection='3d')
  if not hull:
    hull = ConvexHull(c_points)
  for simplex in hull.simplices:
    simplex = np.append(simplex, simplex[0])
    ax.plot(c_points[simplex, 0], c_points[simplex, 1], c_points[simplex, 2], 'k-', linewidth=0.5);
  # write_to_txt(c_pointoints)
  plt.show()

def visualize_s_points(
  s_points: np.ndarray,
  hull: ConvexHull = None,
):
  c_points = to_c_points(s_points)
  visualize_c_points(c_points, hull)

import random
def random_initialize(n):
  s_points = []
  s_points += [[1, 0, pi/2]]
  for i in range(1, n):
    theta = random.random() * 2 * pi
    phi = random.random() * pi - pi/2
    sp = [1, theta, phi]
    s_points += [sp]
      
  s_points = np.array(s_points)
  c_points = to_c_points(s_points)
  return s_points, c_points
