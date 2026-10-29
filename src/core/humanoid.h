#pragma once
// Index layout of the humanoid inside a compiled mjModel, and action mapping.

#include <mujoco/mujoco.h>

#include <string>
#include <vector>

namespace parcour {

// The 18 major body segments (fingers are lumped into the grip and excluded).
inline constexpr const char* kSegmentNames[] = {
    "pelvis",      "abdomen",   "chest",  "head",
    "upper_arm_l", "forearm_l", "hand_l", "upper_arm_r", "forearm_r", "hand_r",
    "thigh_l",     "shank_l",   "foot_l", "toes_l",
    "thigh_r",     "shank_r",   "foot_r", "toes_r"};
inline constexpr int kNumSegments = sizeof(kSegmentNames) / sizeof(kSegmentNames[0]);

class HumanoidLayout {
 public:
  explicit HumanoidLayout(const mjModel* m);

  int root_body = -1;              // pelvis (carries the free joint)
  int root_qpos = -1, root_dof = -1;
  std::vector<int> segments;       // body id per kSegmentNames entry
  std::vector<int> bodies;         // every body of the humanoid (pelvis subtree)
  std::vector<int> hinge_joints;   // every hinge joint, incl. fingers
  std::vector<int> actuated_joints;  // hinge joints driven directly by a motor (no fingers)
  std::vector<int> grip_tendons;     // tendon id per hand (l, r)
  int head_acc_adr = -1, head_gyro_adr = -1;  // sensordata addresses
  int touch_adr[4] = {-1, -1, -1, -1};        // sole_l, toesole_l, sole_r, toesole_r
  double total_mass = 0;

  int num_actions() const { return nu_; }

  // Maps actions in [-1, 1] onto the actuator torque ranges. 0 means zero torque,
  // +1 the full positive limit and -1 the full negative limit (ranges can be asymmetric).
  void apply_action(const mjModel* m, mjData* d, const double* action) const;

  std::string segment_name(int i) const { return kSegmentNames[i]; }
  int segment_of_body(int body) const;  // -1 if not a major segment

 private:
  int nu_ = 0;
  std::vector<int> segment_of_body_;
};

}  // namespace parcour
