#include "core/course.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace parcour {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSlab = 0.4;        // thickness of floor slabs
constexpr double kAbyss = -6.0;      // pillars reach down to the abyss plane
constexpr double kMaxLevel = 2.4;    // floor levels stay within [0, kMaxLevel]
constexpr double kMaxHeading = 75 * kPi / 180;  // keeps the route from curling back

// Portable RNG: std distributions differ between standard libraries, and courses must be
// identical on every machine for a given seed.
struct Rng {
  uint64_t state;
  // hash the seed so neighbouring seeds give unrelated streams
  explicit Rng(uint64_t seed) : state(mix(seed + 0xD1B54A32D192ED03ull)) {}
  static uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint64_t next() { return mix(state += 0x9E3779B97F4A7C15ull); }  // splitmix64
  double uniform(double lo, double hi) { return lo + (hi - lo) * double(next() >> 11) * 0x1.0p-53; }
  int integer(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }  // inclusive
  bool chance(double p) { return uniform(0, 1) < p; }
};

double lerp(double a, double b, double t) { return a + (b - a) * t; }

void yaw_quat(double yaw, double q[4]) {
  q[0] = std::cos(yaw / 2), q[1] = 0, q[2] = 0, q[3] = std::sin(yaw / 2);
}

// Builds the pieces of one section in its local frame (u forward, v left, absolute z).
class SectionBuilder {
 public:
  SectionBuilder(Course& c, int index) : c_(c), sec_(c.sections[index]), index_(index) {}

  void box(Obstacle type, double u0, double u1, double v0, double v1, double z0, double z1) {
    Piece p;
    p.type = type;
    p.section = index_;
    p.shape = mjGEOM_BOX;
    p.pos = sec_.to_world(0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (z0 + z1));
    yaw_quat(sec_.yaw, p.quat);
    p.size = {0.5 * (u1 - u0), 0.5 * (v1 - v0), 0.5 * (z1 - z0)};
    c_.pieces.push_back(p);
  }
  // floor slab across the full corridor width, top at z
  void floor(double u0, double u1, double z, Obstacle type = Obstacle::kFloor) {
    const double w = c_.params.width / 2;
    box(type, u0, u1, -w, w, z - kSlab, z);
  }
  // horizontal bar across the corridor at (u, z)
  void bar(double u, double z, double radius, double half_len) {
    Piece p;
    p.type = Obstacle::kBar;
    p.section = index_;
    p.shape = mjGEOM_CAPSULE;
    p.pos = sec_.to_world(u, 0, z);
    // capsule axis is local z: rotate it onto the section's lateral (v) direction
    double qy[4], qx[4] = {std::cos(-kPi / 4), std::sin(-kPi / 4), 0, 0};
    yaw_quat(sec_.yaw, qy);
    mju_mulQuat(p.quat, qy, qx);
    p.size = {radius, half_len, 0};
    c_.pieces.push_back(p);
  }

 private:
  Course& c_;
  Section& sec_;
  int index_;
};

// Each generator fills in the section's length/levels and adds its pieces. The section
// already has its type, origin, yaw and level_in set.
void make_runway(Course& c, int i, double length) {
  Section& s = c.sections[i];
  s.length = length;
  s.level_out = s.level_min = s.level_in;
  s.obstacle_start = s.obstacle_end = length;
  SectionBuilder(c, i).floor(0, length, s.level_in);
}

void make_stairs(Course& c, int i, Rng& r, double d, double approach, double runout) {
  Section& s = c.sections[i];
  const double rise = r.uniform(0.10, 0.12 + 0.10 * d);
  const double tread = r.uniform(0.28, 0.38);
  int n = r.integer(2, 3 + int(std::lround(5 * d)));
  const bool can_up = s.level_in + 2 * rise <= kMaxLevel, can_down = s.level_in - 2 * rise >= 0;
  const bool up = can_up && (!can_down || s.level_in < 0.3 || r.chance(0.6));
  n = std::min(n, int(std::floor((up ? kMaxLevel - s.level_in : s.level_in) / rise)));
  const double dir = up ? 1 : -1;
  s.level_out = s.level_in + dir * n * rise;
  s.level_min = std::min(s.level_in, s.level_out);
  s.obstacle_start = approach;
  s.obstacle_end = approach + (n - 1) * tread;
  s.length = s.obstacle_end + runout;

  SectionBuilder b(c, i);
  const double w = c.params.width / 2, bottom = s.level_min - kSlab;
  b.floor(0, approach, s.level_in);
  for (int k = 1; k < n; ++k) {  // n rises, n - 1 treads in between
    const double u0 = approach + (k - 1) * tread;
    b.box(Obstacle::kStairs, u0, u0 + tread, -w, w, bottom, s.level_in + dir * k * rise);
  }
  b.floor(s.obstacle_end, s.length, s.level_out);
}

void make_hurdle(Course& c, int i, Rng& r, double d, double approach, double runout) {
  Section& s = c.sections[i];
  const double h = r.uniform(0.15, 0.2 + 0.8 * d);
  const double t = r.chance(0.5) ? r.uniform(0.04, 0.08) : r.uniform(0.3, 0.3 + 0.6 * d);
  s.level_out = s.level_min = s.level_in;
  s.obstacle_start = approach;
  s.obstacle_end = approach + t;
  s.length = s.obstacle_end + runout;

  SectionBuilder b(c, i);
  const double w = c.params.width / 2;
  b.floor(0, s.length, s.level_in);
  b.box(Obstacle::kHurdle, approach, approach + t, -w, w, s.level_in, s.level_in + h);
}

void make_wall(Course& c, int i, Rng& r, double d, double approach, double runout) {
  Section& s = c.sections[i];
  const double w = c.params.width / 2, over = 0.5;  // wall sticks out past the floor
  const double t = r.uniform(0.15, 0.3), height = 2.6;
  const double hole_w = r.uniform(0.65 + 0.35 * (1 - d), 1.2 - 0.2 * d);
  const double hole_h = r.uniform(0.7 + 0.5 * (1 - d), 1.3 - 0.2 * d);
  const double hole_z = r.uniform(0.0, 0.1 + 0.8 * d);
  const double reach = std::max(0.0, w - hole_w / 2 - 0.2);
  const double hole_v = r.uniform(-reach, reach);
  s.level_out = s.level_min = s.level_in;
  s.obstacle_start = approach;
  s.obstacle_end = approach + t;
  s.length = s.obstacle_end + runout;

  SectionBuilder b(c, i);
  const double u0 = approach, u1 = approach + t, z = s.level_in;
  b.floor(0, s.length, z);
  b.box(Obstacle::kWall, u0, u1, hole_v + hole_w / 2, w + over, z, z + height);   // left
  b.box(Obstacle::kWall, u0, u1, -w - over, hole_v - hole_w / 2, z, z + height);  // right
  if (hole_z > 0.01) {
    b.box(Obstacle::kWall, u0, u1, hole_v - hole_w / 2, hole_v + hole_w / 2, z, z + hole_z);
  }
  b.box(Obstacle::kWall, u0, u1, hole_v - hole_w / 2, hole_v + hole_w / 2, z + hole_z + hole_h,
        z + height);
}

void make_beam(Course& c, int i, Rng& r, double d, double approach, double runout) {
  Section& s = c.sections[i];
  const double w = c.params.width / 2;
  const double pit = r.uniform(1.5, 2.0 + 3.0 * d);
  const double beam_w = r.uniform(lerp(0.45, 0.12, d), lerp(0.6, 0.25, d));
  const double reach = 0.5 * std::max(0.0, w - beam_w / 2 - 0.2);
  const double v = r.uniform(-reach, reach);
  s.level_out = s.level_min = s.level_in;
  s.obstacle_start = approach;
  s.obstacle_end = approach + pit;
  s.length = s.obstacle_end + runout;

  SectionBuilder b(c, i);
  b.floor(0, approach, s.level_in);
  b.box(Obstacle::kBeam, approach - 0.05, s.obstacle_end + 0.05, v - beam_w / 2, v + beam_w / 2,
        s.level_in - 0.3, s.level_in);
  b.floor(s.obstacle_end, s.length, s.level_in);
}

void make_bars(Course& c, int i, Rng& r, double d, double approach, double runout) {
  Section& s = c.sections[i];
  const double w = c.params.width / 2;
  const int n = 1 + r.integer(0, int(std::lround(2 * d)));
  const double height = r.uniform(2.0, 2.0 + 0.4 * d);  // above the floor
  const double spacing = r.uniform(1.0, 1.2 + 0.8 * d);
  const double edge = r.uniform(0.5, 0.7 + 0.5 * d);    // pit edge to first / last bar
  const double radius = r.uniform(0.02, 0.03);
  const double pit = (n - 1) * spacing + 2 * edge;
  s.level_out = s.level_min = s.level_in;
  s.obstacle_start = approach;
  s.obstacle_end = approach + pit;
  s.length = s.obstacle_end + runout;

  SectionBuilder b(c, i);
  const double z = s.level_in + height, post_v = w + 0.25;
  b.floor(0, approach, s.level_in);
  for (int k = 0; k < n; ++k) {
    const double u = approach + edge + k * spacing;
    b.bar(u, z, radius, post_v);
    for (double side : {-1.0, 1.0}) {  // posts rise from the abyss outside the corridor
      b.box(Obstacle::kBar, u - 0.05, u + 0.05, side * post_v - 0.05, side * post_v + 0.05,
            kAbyss, z);
    }
  }
  b.floor(s.obstacle_end, s.length, s.level_in);
}

void make_platforms(Course& c, int i, Rng& r, double d, double approach, double runout) {
  Section& s = c.sections[i];
  const double W = c.params.width;
  const int n = 2 + r.integer(0, int(std::lround(2 * d)));
  SectionBuilder b(c, i);
  b.floor(0, approach, s.level_in);
  double u = approach, level = s.level_in, lowest = s.level_in;
  for (int k = 0; k < n; ++k) {
    u += r.uniform(0.2, 0.3 + 1.2 * d);  // gap
    const double len = r.uniform(lerp(1.5, 0.8, d), lerp(2.5, 1.8, d));
    level = std::clamp(level + r.uniform(-0.3 - 0.5 * d, 0.2 + 0.8 * d), 0.0, kMaxLevel);
    const double pw = r.uniform(std::max(0.8, W * (1 - 0.6 * d)), W);
    const double v = r.uniform(-(W - pw) / 2, (W - pw) / 2);
    b.box(Obstacle::kPlatform, u, u + len, v - pw / 2, v + pw / 2, kAbyss, level);
    lowest = std::min(lowest, level);
    u += len;
  }
  s.obstacle_start = approach;
  s.obstacle_end = u + r.uniform(0.2, 0.3 + 1.2 * d);  // last gap
  s.length = s.obstacle_end + runout;
  s.level_out = level;
  s.level_min = lowest;
  b.floor(s.obstacle_end, s.length, level);
}

}  // namespace

const char* obstacle_name(Obstacle t) {
  switch (t) {
    case Obstacle::kFloor: return "floor";
    case Obstacle::kStairs: return "stairs";
    case Obstacle::kHurdle: return "hurdle";
    case Obstacle::kWall: return "wall";
    case Obstacle::kBeam: return "beam";
    case Obstacle::kBar: return "bars";
    case Obstacle::kPlatform: return "platforms";
    default: return "?";
  }
}

Vec3 Section::to_world(double u, double v, double z) const {
  const double c = std::cos(yaw), s = std::sin(yaw);
  return {origin.x + c * u - s * v, origin.y + s * u + c * v, z};
}

Course generate_course(const CourseParams& params) {
  Course c;
  c.params = params;
  Rng r(params.seed);
  const double d = std::clamp(params.difficulty, 0.0, 1.0);

  std::vector<Obstacle> pool;
  for (int t = 1; t < kNumObstacleTypes; ++t) {
    if (params.allowed[t]) pool.push_back(Obstacle(t));
  }
  std::vector<Obstacle> order = {Obstacle::kFloor};
  for (int k = 0; k < params.num_obstacles && !pool.empty(); ++k) {
    Obstacle t = pool[r.integer(0, int(pool.size()) - 1)];
    if (pool.size() > 1 && t == order.back()) t = pool[r.integer(0, int(pool.size()) - 1)];
    order.push_back(t);
  }
  order.push_back(Obstacle::kFloor);

  Vec3 at;
  double yaw = 0, s0 = 0;
  const double max_turn = params.max_turn_deg * kPi / 180;
  for (size_t i = 0; i < order.size(); ++i) {
    if (i > 0) {  // turn at the junction, then cover the outside corner with a round pad
      const double turn = std::clamp(yaw + r.uniform(-max_turn, max_turn), -kMaxHeading,
                                     kMaxHeading) - yaw;
      yaw += turn;
      if (std::fabs(turn) > 0.5 * kPi / 180) {
        Piece pad;
        pad.section = int(i);
        pad.shape = mjGEOM_CYLINDER;
        pad.pos = {at.x, at.y, at.z - 0.002 - kSlab / 2};  // just below the slabs: no z-fighting
        pad.size = {params.width / 2, kSlab / 2, 0};
        c.pieces.push_back(pad);
      }
    }
    Section sec;
    sec.type = order[i];
    sec.origin = at;
    sec.yaw = yaw;
    sec.s0 = s0;
    sec.level_in = at.z;
    c.sections.push_back(sec);
    const int k = int(i);
    const double approach = r.uniform(1.2, 2.0), runout = r.uniform(1.2, 2.0);
    switch (order[i]) {
      case Obstacle::kFloor: make_runway(c, k, i == 0 ? 3.0 : 4.0); break;
      case Obstacle::kStairs: make_stairs(c, k, r, d, approach, runout); break;
      case Obstacle::kHurdle: make_hurdle(c, k, r, d, approach, runout); break;
      case Obstacle::kWall: make_wall(c, k, r, d, approach, runout); break;
      case Obstacle::kBeam: make_beam(c, k, r, d, approach, runout); break;
      case Obstacle::kBar: make_bars(c, k, r, d, approach, runout); break;
      case Obstacle::kPlatform: make_platforms(c, k, r, d, approach, runout); break;
      default: break;
    }
    const Section& done = c.sections.back();
    at = done.to_world(done.length, 0, done.level_out);
    s0 += done.length;
  }
  c.length = s0;
  c.finish_s = s0 - 2.0;  // halfway along the finish runway
  return c;
}

RoutePoint Course::project(const Vec3& p) const {
  RoutePoint best;
  double best_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < sections.size(); ++i) {
    const Section& s = sections[i];
    const double c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const double dx = p.x - s.origin.x, dy = p.y - s.origin.y;
    const double u = c * dx + sn * dy, v = -sn * dx + c * dy;
    const double uc = std::clamp(u, 0.0, s.length);
    const double dist = std::hypot(u - uc, v);
    if (dist < best_dist) {
      best_dist = dist;
      best = {s.s0 + uc, v, int(i)};
    }
  }
  return best;
}

int Course::section_at(double s) const {
  int i = 0;
  while (i + 1 < int(sections.size()) && s >= sections[i + 1].s0) ++i;
  return i;
}

Vec3 Course::point_at(double s) const {
  const Section& sec = sections[section_at(s)];
  const double u = std::clamp(s - sec.s0, 0.0, sec.length);
  // floor height along the centreline is only exact for runways; use the nearer level
  return sec.to_world(u, 0, u < sec.length / 2 ? sec.level_in : sec.level_out);
}

double Course::heading_at(double s) const { return sections[section_at(s)].yaw; }

std::string default_course_scene() { return std::string(PARCOUR_ASSET_DIR) + "/scene_course.xml"; }

mjModel* build_course_model(const mjSpec* base, const Course& course) {
  mjSpec* spec = mj_copySpec(base);
  if (!spec) throw std::runtime_error("mj_copySpec failed");
  mjsBody* world = mjs_findBody(spec, "world");
  // colours per obstacle type; floor slabs use the checker material
  const float rgba[kNumObstacleTypes][4] = {
      {0.52, 0.54, 0.57, 1},           {0.80, 0.72, 0.55, 1}, {0.95, 0.55, 0.15, 1},
      {0.62, 0.36, 0.30, 1},  {0.55, 0.38, 0.22, 1}, {0.75, 0.78, 0.82, 1},
      {0.45, 0.55, 0.68, 1}};
  for (size_t i = 0; i < course.pieces.size(); ++i) {
    const Piece& p = course.pieces[i];
    mjsGeom* g = mjs_addGeom(world, nullptr);
    mjs_setName(g->element, ("course_" + std::to_string(i)).c_str());
    g->type = p.shape;
    p.pos.to(g->pos);
    for (int k = 0; k < 4; ++k) g->quat[k] = p.quat[k];
    p.size.to(g->size);
    g->contype = g->conaffinity = 1;
    g->group = 2;
    if (p.type == Obstacle::kFloor && p.shape == mjGEOM_BOX) mjs_setString(g->material, "floor");
    for (int k = 0; k < 4; ++k) g->rgba[k] = rgba[int(p.type)][k];
  }
  mjModel* m = mj_compile(spec, nullptr);
  const std::string err = m ? "" : mjs_getError(spec);
  mj_deleteSpec(spec);
  if (!m) throw std::runtime_error("course compile failed: " + err);
  return m;
}

int first_course_geom(const mjModel* m) { return mj_name2id(m, mjOBJ_GEOM, "course_0"); }

}  // namespace parcour
