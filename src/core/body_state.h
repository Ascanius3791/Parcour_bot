#pragma once
// Proprioceptive sensors: everything the humanoid "feels" about its own body.

#include <mujoco/mujoco.h>

#include <array>
#include <vector>

#include "core/humanoid.h"
#include "core/injury.h"
#include "core/vec.h"

namespace parcour {

// One major body segment. Vectors use world axes; flatten() rotates them into the heading frame.
struct LimbState {
  Vec3 pos;           // segment com relative to the whole-body com (m)
  Vec3 vel;           // velocity of the segment com (m/s)
  Vec3 ang_vel;       // angular velocity (rad/s)
  Vec3 momentum;      // linear momentum m*v (kg m/s)
  Vec3 ang_momentum;  // angular momentum about the whole-body com (kg m^2/s)
  double rot[9];      // orientation, row-major rotation matrix (body -> world)
};

struct BodyState {
  double time = 0;
  Vec3 com, com_vel;      // whole-body centre of mass and its velocity
  Vec3 momentum;          // total linear momentum
  Vec3 ang_momentum;      // total angular momentum about the com
  double yaw = 0;         // heading of the pelvis around the world z axis
  std::array<LimbState, kNumSegments> limbs;
  std::vector<double> joint_pos, joint_vel;  // actuated hinge joints (rad, rad/s)
  double grip[2] = {0, 0};                   // mean finger flexion per hand (rad)
  double contact[4] = {0, 0, 0, 0};          // sole_l, toes_l, sole_r, toes_r normal force (N)
  Vec3 head_acc, head_gyro;                  // vestibular sense, head frame
  std::array<double, kNumSegments> stress{};  // joint stress, 1 = breaking point

  // Policy input: all quantities expressed in the heading frame (yaw-invariant).
  void flatten(std::vector<float>& out, double body_weight) const;
  static int flat_size(int num_joints);
};

// Requires the position and velocity stages of d to be current (after mj_step1 / mj_forward).
void compute_body_state(const mjModel* m, mjData* d, const HumanoidLayout& layout,
                        const InjuryMonitor* injury, BodyState& out);

}  // namespace parcour
