#include "core/simulation.h"

#include <cmath>
#include <stdexcept>

namespace parcour {

namespace {
mjModel* load(const std::string& xml_path) {
  char err[1024] = "";
  mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
  if (!m) throw std::runtime_error("failed to load " + xml_path + ": " + err);
  return m;
}
}  // namespace

Simulation::Simulation(const std::string& xml_path, InjuryConfig injury)
    : Simulation(load(xml_path), injury) {}

Simulation::Simulation(mjModel* model, InjuryConfig injury) : injury_cfg_(injury) {
  set_model(model);
  reset();
}

Simulation::~Simulation() {
  mj_deleteData(d_);
  mj_deleteModel(m_);
}

void Simulation::set_model(mjModel* model) {
  injury_.reset();
  layout_.reset();
  mj_deleteData(d_);
  mj_deleteModel(m_);
  m_ = model;
  d_ = mj_makeData(m_);
  layout_ = std::make_unique<HumanoidLayout>(m_);
  injury_ = std::make_unique<InjuryMonitor>(m_, *layout_, injury_cfg_);
  state_time_ = -1;
}

void Simulation::reset(double lift) { reset_at({}, 0.0, lift); }

void Simulation::reset_at(const Vec3& feet, double yaw, double lift) {
  mj_resetData(m_, d_);
  double* root = d_->qpos + layout_->root_qpos;
  // the default pose stands on z = 0 at the origin; start just above the floor
  root[0] += feet.x;
  root[1] += feet.y;
  root[2] += feet.z + 0.002 + lift;
  root[3] = std::cos(yaw / 2), root[4] = 0, root[5] = 0, root[6] = std::sin(yaw / 2);
  mj_forward(m_, d_);
  injury_->reset();
  state_time_ = -1;
}

bool Simulation::substep() {
  // Invariant: position/velocity stages of d_ are current. mj_step2 computes the
  // acceleration stage (incl. joint reaction forces) and integrates; the injury monitor
  // reads those forces before mj_step1 recomputes positions for the new state.
  mj_step2(m_, d_);
  injury_->update(m_, d_);
  mj_step1(m_, d_);
  return !broken();
}

bool Simulation::step(const double* action) {
  if (action_mode == ActionMode::kTorque) {
    layout_->apply_action(m_, d_, action);
    for (int k = 0; k < frame_skip; ++k) {
      if (!substep()) return false;
    }
    return true;
  }
  action_.assign(action, action + m_->nu);
  for (int k = 0; k < frame_skip; ++k) {
    layout_->apply_pd_action(m_, d_, action_.data());
    if (!substep()) return false;
  }
  return true;
}

void Simulation::forward() {
  mj_forward(m_, d_);
  state_time_ = -1;
}

const BodyState& Simulation::body_state() {
  if (state_time_ != d_->time) {
    compute_body_state(m_, d_, *layout_, injury_.get(), state_);
    state_time_ = d_->time;
  }
  return state_;
}

}  // namespace parcour
