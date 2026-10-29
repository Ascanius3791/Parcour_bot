#pragma once
// Procedurally generated obstacle corridors and the route through them.
//
// A course is a chain of straight sections: a start runway, N obstacle sections and a finish
// runway. Each section has its own heading (the corridor may turn between sections) and its
// own floor level (stairs and platforms change it). Outside the corridor floor is a void:
// stepping off the side or into a pit is a fall.

#include <mujoco/mujoco.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/vec.h"

namespace parcour {

// Kind of structure a piece belongs to; also the class label the bot's rays will report.
enum class Obstacle : uint8_t {
  kFloor,     // runways, approaches and junction pads
  kStairs,
  kHurdle,    // solid board or vault box across the corridor
  kWall,      // full-height wall with one hole to get through
  kBeam,      // narrow balance beam over a pit
  kBar,       // horizontal bars over a pit, to swing on (and their posts)
  kPlatform,  // pillars separated by gaps, at varying heights
  kCount
};
inline constexpr int kNumObstacleTypes = int(Obstacle::kCount);
const char* obstacle_name(Obstacle t);

struct CourseParams {
  uint64_t seed = 1;
  double difficulty = 0.5;   // 0 = easy versions of every obstacle, 1 = hardest
  int num_obstacles = 6;
  double width = 2.4;        // corridor width (m)
  double max_turn_deg = 25;  // max heading change between sections
  // obstacle types that may appear (kFloor is ignored)
  bool allowed[kNumObstacleTypes] = {false, true, true, true, true, true, true};
};

// One static collision primitive of the course.
struct Piece {
  Obstacle type = Obstacle::kFloor;
  int section = 0;
  mjtGeom shape = mjGEOM_BOX;  // box, capsule (bars) or cylinder (junction pads)
  Vec3 pos;                    // world position of the centre
  double quat[4] = {1, 0, 0, 0};
  Vec3 size;                   // MuJoCo geom size (box: half extents)
};

struct Section {
  Obstacle type = Obstacle::kFloor;
  Vec3 origin;               // start of the section on the centreline, z = entry level
  double yaw = 0;            // heading of the section
  double length = 0;
  double s0 = 0;             // route distance at the section start
  double level_in = 0, level_out = 0;
  double level_min = 0;      // lowest surface you can stand on inside the section
  double obstacle_start = 0; // along-section distance where the obstacle begins (u coordinate)
  double obstacle_end = 0;   // ... and ends

  // world position of a point given in section coordinates (u forward, v left, absolute z)
  Vec3 to_world(double u, double v, double z) const;
};

struct RoutePoint {
  double s = 0;        // distance along the route
  double lateral = 0;  // signed distance from the centreline (left positive)
  int section = 0;
};

struct Course {
  CourseParams params;
  std::vector<Section> sections;
  std::vector<Piece> pieces;
  double length = 0;   // total route length
  double finish_s = 0; // route distance that counts as finished

  RoutePoint project(const Vec3& p) const;   // closest route point to p (xy only)
  Vec3 point_at(double s) const;             // centreline point at route distance s (floor level)
  double heading_at(double s) const;
  int section_at(double s) const;
};

Course generate_course(const CourseParams& params);

// Base scene to which courses are added (humanoid, lights, materials, no floor).
std::string default_course_scene();

// Compiles base + course geoms into a new model (caller owns it). Course piece i becomes the
// geom named "course_<i>"; its id is first_course_geom(model) + i. Throws on failure.
mjModel* build_course_model(const mjSpec* base, const Course& course);
int first_course_geom(const mjModel* m);

}  // namespace parcour
