#include "core/humanoid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace parcour {

namespace {
int require_id(const mjModel* m, mjtObj type, const char* name) {
  const int id = mj_name2id(m, type, name);
  if (id < 0) throw std::runtime_error(std::string("humanoid model is missing '") + name + "'");
  return id;
}

int sensor_adr(const mjModel* m, const char* name) {
  return m->sensor_adr[require_id(m, mjOBJ_SENSOR, name)];
}
}  // namespace

HumanoidLayout::HumanoidLayout(const mjModel* m) : nu_(m->nu) {
  root_body = require_id(m, mjOBJ_BODY, "pelvis");
  const int root_jnt = m->body_jntadr[root_body];
  root_qpos = m->jnt_qposadr[root_jnt];
  root_dof = m->jnt_dofadr[root_jnt];

  for (const char* name : kSegmentNames) segments.push_back(require_id(m, mjOBJ_BODY, name));

  segment_of_body_.assign(m->nbody, -1);
  for (int i = 0; i < kNumSegments; ++i) segment_of_body_[segments[i]] = i;

  // body ids are in depth-first order, so the pelvis subtree is a contiguous id range
  for (int b = root_body; b < m->nbody; ++b) {
    int p = b;
    while (p > 0 && p != root_body) p = m->body_parentid[p];
    if (p != root_body) break;
    bodies.push_back(b);
    total_mass += m->body_mass[b];
  }
  for (int j = 0; j < m->njnt; ++j) {
    if (m->jnt_type[j] == mjJNT_HINGE &&
        std::find(bodies.begin(), bodies.end(), m->jnt_bodyid[j]) != bodies.end()) {
      hinge_joints.push_back(j);
    }
  }

  for (int a = 0; a < m->nu; ++a) {
    if (m->actuator_trntype[a] == mjTRN_JOINT) actuated_joints.push_back(m->actuator_trnid[2 * a]);
  }
  grip_tendons = {require_id(m, mjOBJ_TENDON, "grip_l"), require_id(m, mjOBJ_TENDON, "grip_r")};

  head_acc_adr = sensor_adr(m, "head_acc");
  head_gyro_adr = sensor_adr(m, "head_gyro");
  const char* touch[4] = {"sole_l", "toesole_l", "sole_r", "toesole_r"};
  for (int i = 0; i < 4; ++i) touch_adr[i] = sensor_adr(m, touch[i]);
}

void HumanoidLayout::apply_action(const mjModel* m, mjData* d, const double* action) const {
  for (int i = 0; i < nu_; ++i) {
    const double a = std::clamp(action[i], -1.0, 1.0);
    const double lo = m->actuator_ctrlrange[2 * i], hi = m->actuator_ctrlrange[2 * i + 1];
    d->ctrl[i] = a >= 0 ? a * hi : -a * lo;
  }
}

void HumanoidLayout::apply_pd_action(const mjModel* m, mjData* d, const double* action) const {
  for (int i = 0; i < nu_; ++i) {
    const double a = std::clamp(action[i], -1.0, 1.0);
    const double lo = m->actuator_ctrlrange[2 * i], hi = m->actuator_ctrlrange[2 * i + 1];
    if (m->actuator_trntype[i] != mjTRN_JOINT) {
      d->ctrl[i] = a >= 0 ? a * hi : -a * lo;
      continue;
    }
    const int j = m->actuator_trnid[2 * i];
    const double target = a >= 0 ? a * m->jnt_range[2 * j + 1] : -a * m->jnt_range[2 * j];
    const double kp = pd_kp * std::max(std::fabs(lo), hi), kd = pd_kd * kp;
    const double q = d->qpos[m->jnt_qposadr[j]], qd = d->qvel[m->jnt_dofadr[j]];
    d->ctrl[i] = std::clamp(kp * (target - q) - kd * qd, lo, hi);
  }
}

int HumanoidLayout::segment_of_body(int body) const {
  return body >= 0 && body < static_cast<int>(segment_of_body_.size()) ? segment_of_body_[body]
                                                                        : -1;
}

}  // namespace parcour
