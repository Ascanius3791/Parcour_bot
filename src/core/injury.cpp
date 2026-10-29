#include "core/injury.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/vec.h"

namespace parcour {

namespace {
// joint name and strength for the joint above each segment, matched by segment-name prefix
struct JointSpec {
  const char* segment_prefix;
  const char* joint;
  JointStrength strength;
};

// Rough values for an athletic adult (body weight ~690 N). Tunable.
// {force, off-axis torque, limit torque at lower range end, limit torque at upper range end}
constexpr JointSpec kJointSpecs[] = {
    {"abdomen", "lumbar", {6000, 1e9, 400, 400}},
    {"chest", "thorax", {6000, 1e9, 400, 400}},
    {"head", "neck", {2500, 1e9, 120, 120}},
    {"upper_arm", "shoulder", {3500, 1e9, 300, 300}},
    {"forearm", "elbow", {3000, 150, 200, 1500}},    // hyperextension vs. full flexion
    {"hand", "wrist", {2500, 80, 80, 80}},
    {"thigh", "hip", {10000, 1e9, 500, 500}},
    {"shank", "knee", {9000, 250, 400, 3000}},       // hyperextension vs. deep squat
    {"foot", "ankle", {9000, 250, 300, 300}},
    {"toes", "toe", {5000, 150, 200, 200}},
};

const JointSpec* spec_for(int segment) {
  const char* name = kSegmentNames[segment];
  for (const auto& s : kJointSpecs) {
    if (std::strncmp(name, s.segment_prefix, std::strlen(s.segment_prefix)) == 0) return &s;
  }
  return nullptr;  // pelvis: free joint, cannot break
}
}  // namespace

InjuryConfig InjuryConfig::defaults() {
  InjuryConfig c;
  for (int i = 0; i < kNumSegments; ++i) {
    if (const JointSpec* s = spec_for(i)) c.strength[i] = s->strength;
  }
  return c;
}

std::string InjuryMonitor::joint_name(int segment) {
  const JointSpec* s = spec_for(segment);
  if (!s) return "root";
  std::string name = s->joint;
  const std::string seg = kSegmentNames[segment];
  if (seg.size() > 2 && seg[seg.size() - 2] == '_') name += seg.substr(seg.size() - 2);
  return name;
}

InjuryMonitor::InjuryMonitor(const mjModel* m, const HumanoidLayout& layout, InjuryConfig config)
    : layout_(layout), config_(config) {
  seg_joints_.resize(kNumSegments);
  joint_segment_.assign(m->njnt, -1);
  for (int j = 0; j < m->njnt; ++j) {
    const int seg = layout.segment_of_body(m->jnt_bodyid[j]);
    if (seg > 0 && m->jnt_type[j] == mjJNT_HINGE) {
      seg_joints_[seg].push_back(j);
      joint_segment_[j] = seg;
    }
  }
  reset();
}

void InjuryMonitor::reset() {
  loads_.assign(kNumSegments, JointLoad{});
  raw_lo_.assign(kNumSegments, 0.0);
  raw_hi_.assign(kNumSegments, 0.0);
  broken_ = -1;
}

double InjuryMonitor::max_stress() const {
  double s = 0;
  for (const auto& l : loads_) s = std::max(s, l.stress);
  return s;
}

void InjuryMonitor::update(const mjModel* m, const mjData* d) {
  const double dt = m->opt.timestep;
  const double alpha = dt / (config_.filter_tau + dt);

  // torques of active joint-limit constraints, per segment
  std::fill(raw_lo_.begin(), raw_lo_.end(), 0.0);
  std::fill(raw_hi_.begin(), raw_hi_.end(), 0.0);
  for (int i = 0; i < d->nefc; ++i) {
    if (d->efc_type[i] != mjCNSTR_LIMIT_JOINT) continue;
    const int j = d->efc_id[i];
    const int seg = joint_segment_[j];
    if (seg <= 0) continue;
    const double q = d->qpos[m->jnt_qposadr[j]];
    const bool upper = q > 0.5 * (m->jnt_range[2 * j] + m->jnt_range[2 * j + 1]);
    double& raw = upper ? raw_hi_[seg] : raw_lo_[seg];
    raw = std::max(raw, std::fabs(d->efc_force[i]));
  }

  for (int seg = 1; seg < kNumSegments; ++seg) {
    const int b = layout_.segments[seg];
    const auto& joints = seg_joints_[seg];
    if (joints.empty()) continue;

    // cfrc_int = [torque, force] the parent exerts on this body, about the subtree com
    const double* f6 = d->cfrc_int + 6 * b;
    const Vec3 torque_c = Vec3::from(f6), force = Vec3::from(f6 + 3);
    const Vec3 c = Vec3::from(d->subtree_com + 3 * m->body_rootid[b]);
    const Vec3 anchor = Vec3::from(d->xanchor + 3 * joints[0]);
    Vec3 torque = torque_c + (c - anchor).cross(force);  // shift to the joint anchor

    // remove the components the joint can freely rotate about (Gram-Schmidt on its axes)
    Vec3 basis[3];
    int nb = 0;
    for (int j : joints) {
      Vec3 a = Vec3::from(d->xaxis + 3 * j);
      for (int k = 0; k < nb; ++k) a = a - basis[k] * a.dot(basis[k]);
      const double n = a.norm();
      if (n > 1e-6 && nb < 3) basis[nb++] = a * (1.0 / n);
    }
    for (int k = 0; k < nb; ++k) torque = torque - basis[k] * torque.dot(basis[k]);

    JointLoad& L = loads_[seg];
    L.force += alpha * (force.norm() - L.force);
    L.offaxis += alpha * (torque.norm() - L.offaxis);
    L.limit_lo += alpha * (raw_lo_[seg] - L.limit_lo);
    L.limit_hi += alpha * (raw_hi_[seg] - L.limit_hi);
    const JointStrength& S = config_.strength[seg];
    L.stress = std::max({L.force / S.force, L.offaxis / S.offaxis, L.limit_lo / S.limit_lo,
                         L.limit_hi / S.limit_hi});
    if (L.stress > 1.0 && broken_ < 0) broken_ = seg;
  }
}

}  // namespace parcour
