#include "core/perception.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace parcour {

namespace {
constexpr double kDeg = 3.14159265358979323846 / 180;
constexpr int kCourseGroup = 2;   // geom group of the course pieces
constexpr double kDepthClip = 3;  // piece bottoms deeper than this below the com are clipped

void push(std::vector<float>& out, const Vec3& v) {
  out.push_back(float(v.x));
  out.push_back(float(v.y));
  out.push_back(float(v.z));
}
}  // namespace

Perception::Perception(PerceptionConfig cfg) : cfg_(std::move(cfg)) {
  for (double el : cfg_.elevation_deg) {
    for (double az : cfg_.azimuth_deg) {
      const double ce = std::cos(el * kDeg);
      local_dirs_.push_back({ce * std::cos(az * kDeg), ce * std::sin(az * kDeg), std::sin(el * kDeg)});
    }
  }
  const int n = num_rays();
  dirs_.resize(3 * n);
  dist_.resize(n);
  normal_.resize(3 * n);
  geom_.resize(n);
  hits_.resize(n);
}

void Perception::scan(const mjModel* m, mjData* d, const Course& course, int first_geom,
                      const Vec3& eye, double yaw) {
  eye_ = eye;
  yaw_ = yaw;
  const double c = std::cos(yaw), s = std::sin(yaw);
  const int n = num_rays();
  for (int i = 0; i < n; ++i) {
    const Vec3& l = local_dirs_[i];
    dirs_[3 * i] = c * l.x - s * l.y;
    dirs_[3 * i + 1] = s * l.x + c * l.y;
    dirs_[3 * i + 2] = l.z;
  }
  mjtByte groups[mjNGROUP] = {};
  groups[kCourseGroup] = 1;
  double pnt[3];
  eye.to(pnt);
  mj_multiRay(m, d, pnt, dirs_.data(), groups, 1, -1, geom_.data(), dist_.data(), normal_.data(),
              n, cfg_.max_range);
  const int npieces = int(course.pieces.size());
  for (int i = 0; i < n; ++i) {
    RayHit& h = hits_[i];
    h.dir = Vec3::from(&dirs_[3 * i]);
    const int piece = geom_[i] - first_geom;
    if (geom_[i] >= 0 && dist_[i] >= 0 && dist_[i] <= cfg_.max_range && piece >= 0 && piece < npieces) {
      h.dist = dist_[i];
      h.piece = piece;
      h.type = course.pieces[piece].type;
      h.normal = Vec3::from(&normal_[3 * i]);
    } else {
      h.dist = cfg_.max_range;
      h.piece = -1;
      h.type = Obstacle::kFloor;
      h.normal = {};
    }
  }
}

int Perception::rays_size() const { return num_rays() * (1 + kNumObstacleTypes + 3); }

void Perception::append_rays(std::vector<float>& out) const {
  for (const RayHit& h : hits_) {
    out.push_back(float(h.dist / cfg_.max_range));  // 1 = nothing within range
    for (int t = 0; t < kNumObstacleTypes; ++t) out.push_back(h.piece >= 0 && int(h.type) == t);
    push(out, to_heading(h.normal, yaw_));
  }
}

// per obstacle: present, type one-hot (without kFloor), route distance to start and end,
// heading relative to the body (cos, sin), entry and exit floor level relative to the com,
// and 6 numbers per key piece
int Perception::lookahead_size() const {
  return cfg_.lookahead * (1 + (kNumObstacleTypes - 1) + 2 + 2 + 2 + 6 * cfg_.key_pieces);
}

// lateral offset, heading of the current section relative to the body (cos, sin), centreline
// points ahead, distance to the finish line, time left
int Perception::route_size() const { return 1 + 2 + 3 * int(cfg_.route_ahead.size()) + 2; }

std::vector<int> upcoming_obstacles(const Course& course, double s, int n) {
  std::vector<int> out;
  for (int i = 0; i < int(course.sections.size()) && int(out.size()) < n; ++i) {
    const Section& sec = course.sections[i];
    if (sec.type != Obstacle::kFloor && sec.s0 + sec.obstacle_end > s) out.push_back(i);
  }
  return out;
}

std::vector<int> key_pieces(const Course& course, int section, int max_pieces) {
  std::vector<int> out;
  const Obstacle type = course.sections[section].type;
  for (int i = 0; i < int(course.pieces.size()) && int(out.size()) < max_pieces; ++i) {
    const Piece& p = course.pieces[i];
    if (p.section == section && p.type == type && !p.support) out.push_back(i);
  }
  return out;
}

void append_lookahead(const PerceptionConfig& cfg, const Course& course, const RoutePoint& rp,
                      const Vec3& com, double yaw, std::vector<float>& out) {
  const std::vector<int> next = upcoming_obstacles(course, rp.s, cfg.lookahead);
  for (int k = 0; k < cfg.lookahead; ++k) {
    if (k >= int(next.size())) {
      out.insert(out.end(), 1 + (kNumObstacleTypes - 1) + 2 + 2 + 2 + 6 * cfg.key_pieces, 0.0f);
      continue;
    }
    const Section& sec = course.sections[next[k]];
    out.push_back(1);
    for (int t = 1; t < kNumObstacleTypes; ++t) out.push_back(int(sec.type) == t);
    out.push_back(float(sec.s0 + sec.obstacle_start - rp.s));
    out.push_back(float(sec.s0 + sec.obstacle_end - rp.s));
    out.push_back(float(std::cos(sec.yaw - yaw)));
    out.push_back(float(std::sin(sec.yaw - yaw)));
    out.push_back(float(sec.level_in - com.z));
    out.push_back(float(sec.level_out - com.z));
    // body position in the section frame; piece boxes are given relative to it
    const double c = std::cos(sec.yaw), s = std::sin(sec.yaw);
    const double dx = com.x - sec.origin.x, dy = com.y - sec.origin.y;
    const double u = c * dx + s * dy, v = -s * dx + c * dy;
    const std::vector<int> pieces = key_pieces(course, next[k], cfg.key_pieces);
    for (int j = 0; j < cfg.key_pieces; ++j) {
      if (j >= int(pieces.size())) {
        out.insert(out.end(), 6, 0.0f);
        continue;
      }
      const Piece& p = course.pieces[pieces[j]];
      out.push_back(float(p.lo.x - u));
      out.push_back(float(p.hi.x - u));
      out.push_back(float(p.lo.y - v));
      out.push_back(float(p.hi.y - v));
      out.push_back(float(std::max(p.lo.z - com.z, -kDepthClip)));
      out.push_back(float(p.hi.z - com.z));
    }
  }
}

void append_route(const PerceptionConfig& cfg, const Course& course, const RoutePoint& rp,
                  const Vec3& com, double yaw, double time_left, std::vector<float>& out) {
  const double sec_yaw = course.sections[rp.section].yaw;
  out.push_back(float(rp.lateral));
  out.push_back(float(std::cos(sec_yaw - yaw)));
  out.push_back(float(std::sin(sec_yaw - yaw)));
  for (double ahead : cfg.route_ahead) {
    push(out, to_heading(course.point_at(std::min(rp.s + ahead, course.length)) - com, yaw));
  }
  out.push_back(float(course.finish_s - rp.s));
  out.push_back(float(time_left));
}

}  // namespace parcour
