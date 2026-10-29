#include "core/simulation.h"

#include <stdexcept>

namespace parcour {

Simulation::Simulation(const std::string& xml_path, InjuryConfig injury) {
  char err[1024] = "";
  m_ = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
  if (!m_) throw std::runtime_error("failed to load " + xml_path + ": " + err);
  d_ = mj_makeData(m_);
  layout_ = std::make_unique<HumanoidLayout>(m_);
  injury_ = std::make_unique<InjuryMonitor>(m_, *layout_, injury);
  reset();
}

Simulation::~Simulation() {
  mj_deleteData(d_);
  mj_deleteModel(m_);
}

void Simulation::reset(double lift) {
  mj_resetData(m_, d_);
  d_->qpos[layout_->root_qpos + 2] += 0.002 + lift;  // start just above the floor
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
  layout_->apply_action(m_, d_, action);
  for (int k = 0; k < frame_skip; ++k) {
    if (!substep()) return false;
  }
  return true;
}

const BodyState& Simulation::body_state() {
  if (state_time_ != d_->time) {
    compute_body_state(m_, d_, *layout_, injury_.get(), state_);
    state_time_ = d_->time;
  }
  return state_;
}

}  // namespace parcour
