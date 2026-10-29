#include "core/parkour_env.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace parcour {

const char* end_reason_name(EndReason r) {
  switch (r) {
    case EndReason::kNone: return "running";
    case EndReason::kFinished: return "finished";
    case EndReason::kBroken: return "joint broken";
    case EndReason::kFell: return "fell";
    case EndReason::kDown: return "fell over";
    case EndReason::kOffCourse: return "off course";
    case EndReason::kStalled: return "stalled";
    case EndReason::kTimeout: return "time limit";
  }
  return "?";
}

ParkourEnv::ParkourEnv(EnvConfig cfg, const std::string& base_scene)
    : cfg_(cfg), perception_(cfg.perception), next_seed_(cfg.course.seed),
      noise_state_(cfg.course.seed * 0x9E3779B97F4A7C15ull) {
  char err[1024] = "";
  base_ = mj_parseXML(base_scene.c_str(), nullptr, err, sizeof(err));
  if (!base_) throw std::runtime_error("failed to parse " + base_scene + ": " + err);
  CourseParams p = cfg_.course;
  p.seed = next_seed_++;
  new_course(p);
  reset();
}

ParkourEnv::~ParkourEnv() {
  sim_.reset();
  mj_deleteSpec(base_);
}

void ParkourEnv::new_course(const CourseParams& params) {
  course_ = generate_course(params);
  mjModel* m = build_course_model(base_, course_);
  if (sim_) sim_->set_model(m);
  else sim_ = std::make_unique<Simulation>(m);
  sim_->action_mode = cfg_.action_mode;
  first_geom_ = parcour::first_course_geom(m);
  episodes_on_course_ = 0;
}

void ParkourEnv::reset(double lift, double start_s) {
  if (episodes_on_course_ >= cfg_.episodes_per_course) {
    CourseParams p = cfg_.course;
    p.seed = next_seed_++;
    new_course(p);
  }
  ++episodes_on_course_;
  if (start_s < 0) start_s = cfg_.start_s;
  const Section& start = course_.sections[course_.section_at(start_s)];
  sim_->reset_at(start.to_world(start_s - start.s0, 0, start.level_in), start.yaw, lift);
  if (cfg_.init_noise > 0) {
    const mjModel* m = sim_->model();
    mjData* d = sim_->data();
    for (int j : sim_->layout().actuated_joints) {
      // splitmix64 -> uniform in [-1, 1)
      uint64_t z = (noise_state_ += 0x9E3779B97F4A7C15ull);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      z ^= z >> 31;
      const double u = double(z >> 11) * 0x1.0p-52 - 1.0;
      double& q = d->qpos[m->jnt_qposadr[j]];
      q = std::clamp(q + cfg_.init_noise * u, m->jnt_range[2 * j], m->jnt_range[2 * j + 1]);
    }
    sim_->forward();
  }
  const RoutePoint rp = course_.project(sim_->body_state().com);
  progress_ = best_ = stall_ref_ = rp.s;
  lateral_ = rp.lateral;
  section_ = rp.section;
  stall_since_ = 0;
  finish_time_ = 0;
  reason_ = EndReason::kNone;
}

double ParkourEnv::time() const { return sim_->data()->time; }

double ParkourEnv::control_dt() const { return sim_->frame_skip * sim_->model()->opt.timestep; }

double ParkourEnv::time_limit() const {
  return cfg_.time_limit > 0 ? cfg_.time_limit : 10.0 + 1.5 * course_.length;
}

StepResult ParkourEnv::step(const double* action) {
  if (reason_ != EndReason::kNone) return {0, true, reason_};
  sim_->step(action);
  StepResult r = evaluate();
  const double dt = control_dt();
  r.reward += (cfg_.w_alive - cfg_.w_time) * dt;
  if (cfg_.w_action > 0) {
    const int nu = sim_->model()->nu;
    double sq = 0;
    for (int i = 0; i < nu; ++i) {
      const double a = std::clamp(action[i], -1.0, 1.0);
      sq += a * a;
    }
    r.reward -= cfg_.w_action * sq / nu * dt;
  }
  return r;
}

StepResult ParkourEnv::evaluate() {
  StepResult r;
  if (reason_ != EndReason::kNone) return {0, true, reason_};
  const BodyState& s = sim_->body_state();
  const RoutePoint rp = course_.project(s.com);
  progress_ = rp.s;
  lateral_ = rp.lateral;
  section_ = rp.section;
  if (rp.s > best_) {
    r.reward += cfg_.w_progress * (rp.s - best_);
    best_ = rp.s;
  }
  const double t = time();
  if (best_ > stall_ref_ + 0.25) stall_ref_ = best_, stall_since_ = t;

  const Section& sec = course_.sections[rp.section];
  if (sim_->broken()) reason_ = EndReason::kBroken;
  else if (s.com.z < sec.level_min - cfg_.fall_depth) reason_ = EndReason::kFell;
  else if (cfg_.down_height > 0 && height_above_ground() < cfg_.down_height) reason_ = EndReason::kDown;
  else if (std::abs(rp.lateral) > course_.params.width / 2 + cfg_.lateral_margin)
    reason_ = EndReason::kOffCourse;
  else if (best_ >= course_.finish_s) reason_ = EndReason::kFinished, finish_time_ = t;
  else if (t - stall_since_ > cfg_.stall_time) reason_ = EndReason::kStalled;
  else if (t > time_limit()) reason_ = EndReason::kTimeout;

  if (reason_ == EndReason::kFinished) {
    r.reward += cfg_.finish_bonus + cfg_.finish_time_bonus * std::max(0.0, time_limit() - t);
  }
  if (reason_ == EndReason::kBroken || reason_ == EndReason::kFell ||
      reason_ == EndReason::kDown || reason_ == EndReason::kOffCourse) {
    r.reward -= cfg_.fail_penalty;
  }
  r.done = reason_ != EndReason::kNone;
  r.reason = reason_;
  return r;
}

void ParkourEnv::observe(std::vector<float>& out) {
  const BodyState& s = sim_->body_state();
  s.flatten(out, sim_->layout().total_mass * -sim_->model()->opt.gravity[2]);
  const Vec3 eye = s.com + s.limbs[kHead].pos;
  perception_.scan(sim_->model(), sim_->data(), course_, first_geom_, eye, s.yaw);
  perception_.append_rays(out);
  const RoutePoint rp = course_.project(s.com);
  append_lookahead(cfg_.perception, course_, rp, s.com, s.yaw, out);
  append_route(cfg_.perception, course_, rp, s.com, s.yaw, time_limit() - time(), out);
}

int ParkourEnv::obs_size() const {
  return BodyState::flat_size(int(sim_->layout().actuated_joints.size())) + perception_.size();
}

int ParkourEnv::sections_cleared() const {
  int n = 0;
  for (const Section& s : course_.sections) n += best_ >= s.s0 + s.length ? 1 : 0;
  return n;
}

double ParkourEnv::height_above_ground() {
  const Vec3& com = sim_->body_state().com;
  mjtByte groups[mjNGROUP] = {};
  groups[2] = 1;  // course pieces
  double pnt[3], down[3] = {0, 0, -1};
  com.to(pnt);
  int geom = -1;
  const double h = mj_ray(sim_->model(), sim_->data(), pnt, down, groups, 1, -1, &geom, nullptr);
  return h < 0 ? std::numeric_limits<double>::infinity() : h;
}

double ParkourEnv::score() const {
  return best_ + (finished() ? std::max(0.0, time_limit() - finish_time_) : 0.0);
}

}  // namespace parcour
