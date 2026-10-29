#include "core/body_state.h"

#include <cmath>

namespace parcour {

namespace {
// world-frame inertia tensor of a body applied to w: R diag(I) R^T w
Vec3 inertia_times(const mjModel* m, const mjData* d, int b, const Vec3& w) {
  const double* R = d->ximat + 9 * b;  // row-major, inertial frame -> world
  const double* I = m->body_inertia + 3 * b;
  double local[3];
  for (int i = 0; i < 3; ++i) local[i] = I[i] * (R[i] * w.x + R[3 + i] * w.y + R[6 + i] * w.z);
  return {R[0] * local[0] + R[1] * local[1] + R[2] * local[2],
          R[3] * local[0] + R[4] * local[1] + R[5] * local[2],
          R[6] * local[0] + R[7] * local[1] + R[8] * local[2]};
}
}  // namespace

void compute_body_state(const mjModel* m, mjData* d, const HumanoidLayout& layout,
                        const InjuryMonitor* injury, BodyState& out) {
  mj_subtreeVel(m, d);
  const int root = layout.root_body;
  out.time = d->time;
  out.com = Vec3::from(d->subtree_com + 3 * root);
  out.com_vel = Vec3::from(d->subtree_linvel + 3 * root);
  out.momentum = out.com_vel * layout.total_mass;
  out.ang_momentum = Vec3::from(d->subtree_angmom + 3 * root);
  const double* pr = d->xmat + 9 * root;
  out.yaw = std::atan2(pr[3], pr[0]);

  for (int i = 0; i < kNumSegments; ++i) {
    const int b = layout.segments[i];
    LimbState& L = out.limbs[i];
    double v6[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, b, v6, 0);  // [angular, linear] at the body com
    const double mass = m->body_mass[b];
    L.pos = Vec3::from(d->xipos + 3 * b) - out.com;
    L.ang_vel = Vec3::from(v6);
    L.vel = Vec3::from(v6 + 3);
    L.momentum = L.vel * mass;
    L.ang_momentum = inertia_times(m, d, b, L.ang_vel) + L.pos.cross(L.momentum);
    for (int k = 0; k < 9; ++k) L.rot[k] = d->xmat[9 * b + k];
  }

  const size_t nj = layout.actuated_joints.size();
  out.joint_pos.resize(nj);
  out.joint_vel.resize(nj);
  for (size_t k = 0; k < nj; ++k) {
    const int j = layout.actuated_joints[k];
    out.joint_pos[k] = d->qpos[m->jnt_qposadr[j]];
    out.joint_vel[k] = d->qvel[m->jnt_dofadr[j]];
  }
  for (int h = 0; h < 2; ++h) {
    const int t = layout.grip_tendons[h];
    out.grip[h] = d->ten_length[t] / m->tendon_num[t];
  }
  for (int k = 0; k < 4; ++k) out.contact[k] = d->sensordata[layout.touch_adr[k]];
  out.head_acc = Vec3::from(d->sensordata + layout.head_acc_adr);
  out.head_gyro = Vec3::from(d->sensordata + layout.head_gyro_adr);
  for (int i = 0; i < kNumSegments; ++i) out.stress[i] = injury ? injury->loads()[i].stress : 0.0;
}

int BodyState::flat_size(int num_joints) {
  return 1 + 3 + 3 + 3 + kNumSegments * 18 + 2 * num_joints + 2 + 4 + 6 + kNumSegments;
}

void BodyState::flatten(std::vector<float>& out, double body_weight) const {
  out.clear();
  auto push = [&](const Vec3& v) {
    out.push_back(float(v.x));
    out.push_back(float(v.y));
    out.push_back(float(v.z));
  };
  auto push_h = [&](const Vec3& v) { push(to_heading(v, yaw)); };

  out.push_back(float(com.z));
  push_h(com_vel);
  push_h(momentum);
  push_h(ang_momentum);
  for (const LimbState& L : limbs) {
    push_h(L.pos);
    push_h(L.ang_vel);
    push_h(L.momentum);  // velocity is implied: momentum / constant segment mass
    push_h(L.ang_momentum);
    // first two columns of the rotation matrix: a continuous 6D orientation encoding
    push_h({L.rot[0], L.rot[3], L.rot[6]});
    push_h({L.rot[1], L.rot[4], L.rot[7]});
  }
  for (double q : joint_pos) out.push_back(float(q));
  for (double v : joint_vel) out.push_back(float(v));
  out.push_back(float(grip[0]));
  out.push_back(float(grip[1]));
  for (double c : contact) out.push_back(float(c / body_weight));
  push(head_acc);
  push(head_gyro);
  for (double s : stress) out.push_back(float(s));
}

}  // namespace parcour
