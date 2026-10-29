#pragma once
// Simple hand-written controllers for testing and the viewer (not the learned policy).

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>

#include "core/humanoid.h"

namespace parcour {

// PD controller that tries to hold the default standing pose, clamped to the human torque
// limits. grip in [0, 1] closes the hands.
inline void hold_pose_ctrl(const mjModel* m, mjData* d, double grip) {
  for (int a = 0; a < m->nu; ++a) {
    const double lo = m->actuator_ctrlrange[2 * a], hi = m->actuator_ctrlrange[2 * a + 1];
    if (m->actuator_trntype[a] == mjTRN_TENDON) {
      d->ctrl[a] = grip > 0 ? grip * hi : 0.2 * lo;
      continue;
    }
    const int j = m->actuator_trnid[2 * a];
    const double kp = 5.0 * std::max(std::fabs(lo), hi);
    const double kd = 0.02 * kp;  // explicit damping: higher values chatter at dt = 2 ms
    const double q = d->qpos[m->jnt_qposadr[j]], qd = d->qvel[m->jnt_dofadr[j]];
    d->ctrl[a] = std::clamp(-kp * q - kd * qd, lo, hi);
  }
}

}  // namespace parcour
