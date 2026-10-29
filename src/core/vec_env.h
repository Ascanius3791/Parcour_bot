#pragma once
// A batch of ParkourEnvs stepped in parallel on a pool of threads, for training.
//
// Every env runs its own sequence of courses. Finished episodes restart automatically; the
// observation returned for them is the first one of the new episode, and the last one of the
// old episode is kept in final_obs (needed to bootstrap the value of time-limit endings).

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "core/parkour_env.h"

namespace parcour {

// Runs f(0) ... f(n-1) on worker threads plus the calling thread; blocks until all are done.
class ThreadPool {
 public:
  explicit ThreadPool(int threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  int size() const { return int(workers_.size()) + 1; }
  void run(int n, const std::function<void(int)>& f);

 private:
  void work();
  void drain();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* job_ = nullptr;
  int n_ = 0, next_ = 0, busy_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

struct EpisodeStats {
  int env = 0;
  double ret = 0;        // sum of rewards
  int steps = 0;         // control steps
  double time = 0;       // simulated seconds
  double time_limit = 0;
  double progress = 0;   // best route distance reached, from the start position
  double course_length = 0;  // route distance from the start to the finish line
  double score = 0;
  int sections_cleared = 0;
  EndReason reason = EndReason::kNone;
};

class VecEnv {
 public:
  // Env i starts on course seed cfg.course.seed + i * 1000003.
  VecEnv(const EnvConfig& cfg, int num_envs, int threads);

  int num_envs() const { return int(envs_.size()); }
  int obs_size() const { return obs_size_; }
  int act_size() const { return act_size_; }
  int threads() const { return pool_.size(); }
  ParkourEnv& env(int i) { return *envs_[i]; }

  // Resets every env and writes the observations, [num_envs x obs_size].
  void reset(float* obs);
  // actions [num_envs x act_size] in [-1, 1]. Writes obs, reward, done (episode ended),
  // truncated (ended by the time limit) and, for envs that are done, final_obs (row-wise).
  void step(const float* actions, float* obs, float* reward, uint8_t* done, uint8_t* truncated,
            float* final_obs);
  // Episodes finished since the last call.
  std::vector<EpisodeStats> take_episodes();

 private:
  EnvConfig cfg_;
  std::vector<std::unique_ptr<ParkourEnv>> envs_;
  std::vector<std::vector<float>> obs_buf_;
  std::vector<std::vector<double>> act_buf_;
  std::vector<double> ret_;
  std::vector<int> steps_;
  std::vector<double> start_;  // route distance where the current episode started
  std::vector<EpisodeStats> finished_;
  std::mutex finished_mu_;
  int obs_size_ = 0, act_size_ = 0;
  ThreadPool pool_;
};

}  // namespace parcour
