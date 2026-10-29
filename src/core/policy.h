#pragma once
// A trained policy exported by train/ (policy.bin), evaluated in C++ for the viewer and tests.
//
// File format: text header lines "key value..." up to a line "end", then little-endian
// float32 data: observation mean[obs], std[obs], then per layer W[out][in] and b[out].
// Required keys: "parcour_policy 1", "layers n0 n1 ... nk" (n0 = obs size, nk = action
// size), "activation elu", "obs_clip c". Other keys describe how it was trained (stage,
// course and env settings) and are kept in meta().

#include <map>
#include <string>
#include <vector>

namespace parcour {

class Policy {
 public:
  explicit Policy(const std::string& path);

  int obs_size() const { return sizes_.front(); }
  int act_size() const { return sizes_.back(); }
  const std::map<std::string, std::string>& meta() const { return meta_; }
  // meta value as a number, or fallback if the key is missing
  double meta_num(const std::string& key, double fallback) const;

  // Deterministic action (the mean of the trained distribution), clamped to [-1, 1].
  void act(const std::vector<float>& obs, double* action) const;

 private:
  std::vector<int> sizes_;
  std::vector<float> mean_, std_;
  std::vector<std::vector<float>> w_, b_;
  float clip_ = 10;
  std::map<std::string, std::string> meta_;
  mutable std::vector<float> x_, y_;
};

}  // namespace parcour
