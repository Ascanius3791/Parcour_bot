#pragma once
// Exteroception: what the humanoid perceives of the course around it.
//
//  - Rays cast from the head in a fixed fan relative to the heading. Each ray reports the
//    distance to the first course surface it hits, which kind of obstacle that surface belongs
//    to and the surface normal (its angle). The hit point follows from distance and direction.
//  - The next few obstacles on the route, with their type, where they start and end, their
//    heading, floor levels and the boxes of their key pieces (hurdle board, wall parts around
//    the hole, beam, bars, platform tops, steps).
//  - Route hints: the centreline a few metres ahead, lateral offset, distance and time left.
//
// All vectors are expressed in the heading frame (x forward, y left, z up) so the input does
// not depend on which way the course happens to point.

#include <mujoco/mujoco.h>

#include <vector>

#include "core/course.h"
#include "core/vec.h"

namespace parcour {

struct PerceptionConfig {
  // ray fan: every azimuth (left positive) is combined with every elevation (up positive)
  std::vector<double> azimuth_deg = {-60, -40, -25, -12, 0, 12, 25, 40, 60};
  std::vector<double> elevation_deg = {50, 25, 10, 0, -8, -16, -25, -35, -48, -62, -78};
  double max_range = 10.0;  // metres; rays that hit nothing report this distance
  int lookahead = 3;        // upcoming obstacles listed
  int key_pieces = 4;       // pieces described per upcoming obstacle
  std::vector<double> route_ahead = {1.0, 2.0, 4.0};  // centreline points this far ahead (m)
};

struct RayHit {
  double dist = 0;   // metres, max_range if nothing was hit
  int piece = -1;    // index into Course::pieces, -1 if nothing was hit
  Obstacle type = Obstacle::kFloor;
  Vec3 dir;          // world direction (unit)
  Vec3 normal;       // world surface normal at the hit (zero if nothing was hit)
};

class Perception {
 public:
  explicit Perception(PerceptionConfig cfg = {});

  const PerceptionConfig& config() const { return cfg_; }
  int num_rays() const { return int(local_dirs_.size()); }

  // Casts the ray fan from `eye` with the fan turned to heading `yaw`. Only course geoms
  // (piece i = geom first_geom + i) are seen; the body itself and the abyss are not.
  void scan(const mjModel* m, mjData* d, const Course& course, int first_geom, const Vec3& eye,
            double yaw);
  const std::vector<RayHit>& hits() const { return hits_; }
  const Vec3& eye() const { return eye_; }

  // Policy input (appended to out). Sizes are fixed for a given config.
  void append_rays(std::vector<float>& out) const;  // uses the yaw of the last scan
  int rays_size() const;
  int lookahead_size() const;
  int route_size() const;
  int size() const { return rays_size() + lookahead_size() + route_size(); }

 private:
  PerceptionConfig cfg_;
  std::vector<Vec3> local_dirs_;  // heading frame
  std::vector<double> dirs_, dist_, normal_;
  std::vector<int> geom_;
  std::vector<RayHit> hits_;
  Vec3 eye_;
  double yaw_ = 0;
};

// Section indices of the next n obstacles whose end the route position `s` has not passed.
std::vector<int> upcoming_obstacles(const Course& course, double s, int n);

// Pieces that describe obstacle section `section` (no floors, no supports), along the route.
std::vector<int> key_pieces(const Course& course, int section, int max_pieces);

// Appends the upcoming-obstacle list for a body at route point rp with com and heading yaw.
void append_lookahead(const PerceptionConfig& cfg, const Course& course, const RoutePoint& rp,
                      const Vec3& com, double yaw, std::vector<float>& out);

// Appends route hints; time_left in seconds.
void append_route(const PerceptionConfig& cfg, const Course& course, const RoutePoint& rp,
                  const Vec3& com, double yaw, double time_left, std::vector<float>& out);

}  // namespace parcour
