#include "core/policy.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace parcour {

Policy::Policy(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open policy " + path);
  std::string line;
  while (std::getline(in, line) && line != "end") {
    const size_t sp = line.find(' ');
    meta_[line.substr(0, sp)] = sp == std::string::npos ? "" : line.substr(sp + 1);
  }
  if (meta_["parcour_policy"] != "1") throw std::runtime_error(path + " is not a parcour policy (v1)");
  if (meta_["activation"] != "elu") throw std::runtime_error("unsupported activation in " + path);
  std::istringstream layers(meta_["layers"]);
  for (int n; layers >> n;) sizes_.push_back(n);
  if (sizes_.size() < 2) throw std::runtime_error("bad layer list in " + path);
  clip_ = float(meta_num("obs_clip", 10));

  auto read = [&](std::vector<float>& v, size_t n) {
    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(float)));
    if (!in) throw std::runtime_error("truncated policy file " + path);
  };
  read(mean_, sizes_[0]);
  read(std_, sizes_[0]);
  for (size_t l = 0; l + 1 < sizes_.size(); ++l) {
    w_.emplace_back();
    b_.emplace_back();
    read(w_.back(), size_t(sizes_[l]) * sizes_[l + 1]);
    read(b_.back(), sizes_[l + 1]);
  }
}

double Policy::meta_num(const std::string& key, double fallback) const {
  const auto it = meta_.find(key);
  return it == meta_.end() || it->second.empty() ? fallback : std::stod(it->second);
}

void Policy::act(const std::vector<float>& obs, double* action) const {
  if (int(obs.size()) != obs_size()) throw std::runtime_error("observation size does not match the policy");
  x_.resize(obs.size());
  for (size_t i = 0; i < obs.size(); ++i) {
    x_[i] = std::clamp((obs[i] - mean_[i]) / std_[i], -clip_, clip_);
  }
  for (size_t l = 0; l < w_.size(); ++l) {
    const int nin = sizes_[l], nout = sizes_[l + 1];
    y_.assign(b_[l].begin(), b_[l].end());
    for (int o = 0; o < nout; ++o) {
      const float* row = &w_[l][size_t(o) * nin];
      float sum = 0;
      for (int i = 0; i < nin; ++i) sum += row[i] * x_[i];
      y_[o] += sum;
    }
    if (l + 1 < w_.size()) {
      for (float& v : y_) v = v > 0 ? v : std::expm1(v);  // elu
    }
    x_.swap(y_);
  }
  for (int i = 0; i < act_size(); ++i) action[i] = std::clamp(double(x_[i]), -1.0, 1.0);
}

}  // namespace parcour
