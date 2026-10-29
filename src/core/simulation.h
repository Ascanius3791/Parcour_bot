#pragma once
// One simulated humanoid in one world: physics stepping, injury monitoring and sensors.

#include <mujoco/mujoco.h>

#include <memory>
#include <string>

#include "core/body_state.h"
#include "core/humanoid.h"
#include "core/injury.h"

namespace parcour {

class Simulation {
 public:
  explicit Simulation(const std::string& xml_path, InjuryConfig injury = InjuryConfig::defaults());
  // Takes ownership of a compiled model.
  explicit Simulation(mjModel* model, InjuryConfig injury = InjuryConfig::defaults());
  ~Simulation();
  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;

  const mjModel* model() const { return m_; }
  mjData* data() { return d_; }
  const HumanoidLayout& layout() const { return *layout_; }
  const InjuryMonitor& injury() const { return *injury_; }

  int frame_skip = 10;  // physics substeps per control step (0.002 s * 10 = 50 Hz control)

  // Replaces the world (e.g. a new obstacle course); takes ownership. Call reset() afterwards.
  void set_model(mjModel* model);

  // Standing default pose, lifted by `lift` metres, at rest.
  void reset(double lift = 0.0);
  // Same, with the feet on the point `feet` and facing `yaw`.
  void reset_at(const Vec3& feet, double yaw, double lift = 0.0);

  // Applies a policy action in [-1, 1]^nu for one control period.
  // Returns false as soon as a joint breaks (the episode is over).
  bool step(const double* action);

  // Advances one physics substep using whatever is in d->ctrl / d->xfrc_applied.
  // Returns false if a joint is broken.
  bool substep();

  bool broken() const { return injury_->broken_segment() >= 0; }

  // Sensors of the current state (recomputed on demand).
  const BodyState& body_state();

 private:
  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  std::unique_ptr<HumanoidLayout> layout_;
  std::unique_ptr<InjuryMonitor> injury_;
  InjuryConfig injury_cfg_;
  BodyState state_;
  double state_time_ = -1;
};

}  // namespace parcour
