#pragma once
// One training environment: a humanoid running through generated obstacle courses, with
// progress scoring and episode termination.

#include <memory>
#include <string>

#include "core/course.h"
#include "core/perception.h"
#include "core/simulation.h"

namespace parcour {

enum class EndReason { kNone, kFinished, kBroken, kFell, kDown, kOffCourse, kStalled, kTimeout };
inline constexpr int kNumEndReasons = 8;
const char* end_reason_name(EndReason r);

struct EnvConfig {
  CourseParams course;           // course.seed is the seed of the first course
  int episodes_per_course = 8;   // compiling a course takes ~20 ms, so reuse it a few times
  double start_s = 1.0;          // start position along the route
  double random_start = 0;       // chance that an episode starts in front of a random obstacle
  ActionMode action_mode = ActionMode::kTarget;
  double init_noise = 0;         // random offset of every joint angle at reset (rad)

  // termination
  double fall_depth = 0.6;       // com this far below the lowest floor of its section = fell
  double down_height = 0;        // com lower than this above the surface below it = down (0: off)
  double lateral_margin = 0.6;   // com this far outside the corridor = off course
  double stall_time = 5.0;       // no new 0.25 m of progress for this long = stalled
  double time_limit = 0;         // 0: 10 s + 1.5 s per metre of route

  // reward per control step: w_progress * new progress (m) + (w_alive - w_time) * dt
  // - w_action * mean(action^2) * dt, plus finish_bonus + finish_time_bonus * seconds left
  // when finishing and - fail_penalty on fell / down / broken / off course
  double w_progress = 1.0;
  double w_alive = 0.0;
  double w_time = 0.0;
  double w_action = 0.0;
  double finish_bonus = 10.0;
  double finish_time_bonus = 0.0;
  double fail_penalty = 0.0;

  PerceptionConfig perception;
};

struct StepResult {
  double reward = 0;
  bool done = false;
  EndReason reason = EndReason::kNone;
};

class ParkourEnv {
 public:
  explicit ParkourEnv(EnvConfig cfg = {}, const std::string& base_scene = default_course_scene());
  ~ParkourEnv();
  ParkourEnv(const ParkourEnv&) = delete;
  ParkourEnv& operator=(const ParkourEnv&) = delete;

  // Starts a new episode; generates the next course every episodes_per_course episodes.
  // start_s < 0 uses config().start_s, or with chance config().random_start the approach of a
  // random obstacle; it must lie on flat floor (a runway or an approach).
  void reset(double lift = 0.0, double start_s = -1);
  // Switches to a specific course now (the next reset() starts on it).
  void new_course(const CourseParams& params);
  // Applies an action for one control step and evaluates the episode.
  StepResult step(const double* action);
  // Updates progress and checks termination for the current state; step() calls this.
  // Useful when the simulation is advanced by other means (the viewer).
  StepResult evaluate();

  // Policy input for the current state: body sensors (BodyState::flatten), ray scan,
  // upcoming obstacles and route hints, in that order. Casts the rays.
  void observe(std::vector<float>& out);
  int obs_size() const;

  Simulation& sim() { return *sim_; }
  const Course& course() const { return course_; }
  const EnvConfig& config() const { return cfg_; }
  int first_course_geom() const { return first_geom_; }  // course piece i = this geom id + i
  const Perception& perception() const { return perception_; }  // as of the last observe()

  double time() const;
  double control_dt() const;
  double time_limit() const;
  double progress() const { return progress_; }          // current route distance of the com
  double best_progress() const { return best_; }         // furthest point reached
  double lateral() const { return lateral_; }
  int section() const { return section_; }
  int sections_cleared() const;
  bool finished() const { return reason_ == EndReason::kFinished; }
  double finish_time() const { return finish_time_; }  // episode time at the finish line
  EndReason end_reason() const { return reason_; }
  // Ranking score: metres reached, plus one point per second left when finished.
  double score() const;
  // Height of the com above the course surface below it (infinity over a pit).
  double height_above_ground();

 private:
  EnvConfig cfg_;
  mjSpec* base_ = nullptr;
  std::unique_ptr<Simulation> sim_;
  Course course_;
  int first_geom_ = -1;
  Perception perception_;
  double uniform();  // in [0, 1), from noise_state_

  int episodes_on_course_ = 0;
  uint64_t next_seed_ = 0;
  uint64_t noise_state_ = 0;

  double progress_ = 0, best_ = 0, lateral_ = 0;
  double stall_ref_ = 0, stall_since_ = 0, finish_time_ = 0;
  int section_ = 0;
  EndReason reason_ = EndReason::kNone;
};

}  // namespace parcour
