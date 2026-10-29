#include "core/vec_env.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace parcour {

ThreadPool::ThreadPool(int threads) {
  for (int i = 1; i < std::max(threads, 1); ++i) workers_.emplace_back([this] { work(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  wake_.notify_all();
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::run(int n, const std::function<void(int)>& f) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    job_ = &f;
    n_ = n;
    next_ = 0;
    ++generation_;
  }
  wake_.notify_all();
  drain();
  std::unique_lock<std::mutex> lk(mu_);
  done_.wait(lk, [&] { return next_ >= n_ && busy_ == 0; });
  job_ = nullptr;
}

// Takes items of the current job until none are left.
void ThreadPool::drain() {
  for (;;) {
    int i;
    const std::function<void(int)>* f;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (next_ >= n_) return;
      i = next_++;
      ++busy_;
      f = job_;
    }
    (*f)(i);
    std::lock_guard<std::mutex> lk(mu_);
    if (--busy_ == 0 && next_ >= n_) done_.notify_all();
  }
}

void ThreadPool::work() {
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      wake_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    drain();
  }
}

VecEnv::VecEnv(const EnvConfig& cfg, int num_envs, int threads)
    : cfg_(cfg), pool_(std::min(threads, num_envs)) {
  if (num_envs < 1) throw std::invalid_argument("VecEnv needs at least one env");
  envs_.resize(num_envs);
  pool_.run(num_envs, [&](int i) {  // compiling the courses is the slow part: in parallel
    EnvConfig c = cfg_;
    c.course.seed = cfg_.course.seed + uint64_t(i) * 1000003;
    envs_[i] = std::make_unique<ParkourEnv>(c);
  });
  obs_size_ = envs_[0]->obs_size();
  act_size_ = envs_[0]->sim().model()->nu;
  obs_buf_.resize(num_envs);
  act_buf_.assign(num_envs, std::vector<double>(act_size_));
  ret_.assign(num_envs, 0.0);
  steps_.assign(num_envs, 0);
  start_.assign(num_envs, 0.0);
}

void VecEnv::reset(float* obs) {
  pool_.run(num_envs(), [&](int i) {
    ParkourEnv& e = *envs_[i];
    e.reset();
    ret_[i] = 0;
    steps_[i] = 0;
    start_[i] = e.best_progress();
    e.observe(obs_buf_[i]);
    std::memcpy(obs + size_t(i) * obs_size_, obs_buf_[i].data(), sizeof(float) * obs_size_);
  });
}

void VecEnv::step(const float* actions, float* obs, float* reward, uint8_t* done,
                  uint8_t* truncated, float* final_obs) {
  pool_.run(num_envs(), [&](int i) {
    ParkourEnv& e = *envs_[i];
    std::vector<double>& a = act_buf_[i];
    for (int k = 0; k < act_size_; ++k) a[k] = actions[size_t(i) * act_size_ + k];
    const StepResult r = e.step(a.data());
    ret_[i] += r.reward;
    ++steps_[i];
    reward[i] = float(r.reward);
    done[i] = r.done;
    truncated[i] = r.reason == EndReason::kTimeout;
    if (r.done) {
      e.observe(obs_buf_[i]);
      std::memcpy(final_obs + size_t(i) * obs_size_, obs_buf_[i].data(), sizeof(float) * obs_size_);
      EpisodeStats st;
      st.env = i;
      st.ret = ret_[i];
      st.steps = steps_[i];
      st.time = e.time();
      st.time_limit = e.time_limit();
      st.progress = e.best_progress() - start_[i];
      st.course_length = e.course().finish_s - start_[i];
      st.score = e.score();
      st.sections_cleared = e.sections_cleared();
      st.reason = r.reason;
      {
        std::lock_guard<std::mutex> lk(finished_mu_);
        finished_.push_back(st);
      }
      e.reset();
      ret_[i] = 0;
      steps_[i] = 0;
      start_[i] = e.best_progress();
    }
    e.observe(obs_buf_[i]);
    std::memcpy(obs + size_t(i) * obs_size_, obs_buf_[i].data(), sizeof(float) * obs_size_);
  });
}

std::vector<EpisodeStats> VecEnv::take_episodes() {
  std::lock_guard<std::mutex> lk(finished_mu_);
  std::vector<EpisodeStats> out;
  out.swap(finished_);
  std::sort(out.begin(), out.end(), [](const EpisodeStats& a, const EpisodeStats& b) { return a.env < b.env; });
  return out;
}

}  // namespace parcour
