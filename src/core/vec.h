#pragma once
// Minimal 3D vector helpers used across the core library.

#include <cmath>

namespace parcour {

struct Vec3 {
  double x = 0, y = 0, z = 0;

  static Vec3 from(const double* p) { return {p[0], p[1], p[2]}; }
  void to(double* p) const { p[0] = x; p[1] = y; p[2] = z; }

  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
  double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3& o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  double norm() const { return std::sqrt(dot(*this)); }
};

// Rotation about the world z axis by -yaw: expresses a world vector in the heading frame.
inline Vec3 to_heading(const Vec3& v, double yaw) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  return {c * v.x + s * v.y, -s * v.x + c * v.y, v.z};
}

}  // namespace parcour
