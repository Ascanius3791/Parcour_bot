#pragma once
// Joint injury model: every joint between two major segments breaks when its load is too high.
//
// Three load measures are tracked per joint (all low-pass filtered, so a single-substep
// contact spike does not count, but a hard landing does):
//   force    - magnitude of the reaction force transmitted through the joint (N)
//   offaxis  - torque about axes the joint cannot rotate around, e.g. bending a knee
//              sideways (Nm); zero for 3-axis joints such as hip and shoulder
//   limit    - torque pushing the joint past its range of motion (Nm), with separate strengths
//              for the lower end (e.g. knee hyperextension, dangerous) and the upper end
//              (e.g. full knee flexion, cushioned by soft tissue in a real body)
// stress = max(load / strength) over the three measures; stress > 1 means the joint broke.

#include <mujoco/mujoco.h>

#include <array>
#include <string>
#include <vector>

#include "core/humanoid.h"

namespace parcour {

struct JointStrength {
  double force = 1e9;
  double offaxis = 1e9;
  double limit_lo = 1e9;
  double limit_hi = 1e9;
};

struct InjuryConfig {
  double filter_tau = 0.01;  // s, time constant of the load low-pass filter
  std::array<JointStrength, kNumSegments> strength{};  // indexed like kSegmentNames
  static InjuryConfig defaults();
};

struct JointLoad {
  double force = 0, offaxis = 0;      // filtered loads
  double limit_lo = 0, limit_hi = 0;  // filtered limit torque at the lower / upper range end
  double stress = 0;                  // max ratio load / strength
};

class InjuryMonitor {
 public:
  InjuryMonitor(const mjModel* m, const HumanoidLayout& layout,
                InjuryConfig config = InjuryConfig::defaults());

  void reset();

  // Reads the joint loads of the last acceleration stage. Call after mj_step2 (or
  // mj_forward), before mj_step1 recomputes positions for the next state.
  void update(const mjModel* m, const mjData* d);

  // Loads per segment (the joint connecting it to its parent); the pelvis entry stays 0.
  const std::vector<JointLoad>& loads() const { return loads_; }
  int broken_segment() const { return broken_; }  // -1 while intact
  double max_stress() const;
  static std::string joint_name(int segment);  // "knee_l", "neck", ...

 private:
  const HumanoidLayout& layout_;
  InjuryConfig config_;
  std::vector<std::vector<int>> seg_joints_;  // hinge joint ids per segment
  std::vector<int> joint_segment_;            // segment per joint id, -1 if none
  std::vector<JointLoad> loads_;
  std::vector<double> raw_lo_, raw_hi_;
  int broken_ = -1;
};

}  // namespace parcour
