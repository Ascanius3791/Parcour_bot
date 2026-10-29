// Python module `parcour`: the batched training environment and policy evaluation, for the
// PyTorch trainer in train/. All heavy work runs in C++ with the GIL released.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/body_state.h"
#include "core/policy.h"
#include "core/vec_env.h"

namespace py = pybind11;
using namespace parcour;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::dict episode_dict(const EpisodeStats& e) {
  py::dict d;
  d["env"] = e.env;
  d["return"] = e.ret;
  d["steps"] = e.steps;
  d["time"] = e.time;
  d["time_limit"] = e.time_limit;
  d["progress"] = e.progress;
  d["course_length"] = e.course_length;
  d["score"] = e.score;
  d["sections_cleared"] = e.sections_cleared;
  d["reason"] = end_reason_name(e.reason);
  d["finished"] = e.reason == EndReason::kFinished;
  return d;
}

}  // namespace

PYBIND11_MODULE(parcour, m) {
  m.doc() = "Parkour humanoid simulation (MuJoCo) for reinforcement learning";

  py::enum_<ActionMode>(m, "ActionMode")
      .value("torque", ActionMode::kTorque)
      .value("target", ActionMode::kTarget);

  py::class_<CourseParams>(m, "CourseParams")
      .def(py::init<>())
      .def_readwrite("seed", &CourseParams::seed)
      .def_readwrite("difficulty", &CourseParams::difficulty)
      .def_readwrite("num_obstacles", &CourseParams::num_obstacles)
      .def_readwrite("width", &CourseParams::width)
      .def_readwrite("max_turn_deg", &CourseParams::max_turn_deg)
      .def_readwrite("finish_length", &CourseParams::finish_length)
      .def_property(
          "allowed",  // obstacle type names that may appear
          [](const CourseParams& p) {
            std::vector<std::string> out;
            for (int t = 1; t < kNumObstacleTypes; ++t) {
              if (p.allowed[t]) out.push_back(obstacle_name(Obstacle(t)));
            }
            return out;
          },
          [](CourseParams& p, const std::vector<std::string>& names) {
            for (int t = 1; t < kNumObstacleTypes; ++t) {
              p.allowed[t] = std::find(names.begin(), names.end(), obstacle_name(Obstacle(t))) != names.end();
            }
          });

  py::class_<EnvConfig>(m, "EnvConfig")
      .def(py::init<>())
      .def_readwrite("course", &EnvConfig::course)
      .def_readwrite("episodes_per_course", &EnvConfig::episodes_per_course)
      .def_readwrite("start_s", &EnvConfig::start_s)
      .def_readwrite("random_start", &EnvConfig::random_start)
      .def_readwrite("action_mode", &EnvConfig::action_mode)
      .def_readwrite("init_noise", &EnvConfig::init_noise)
      .def_readwrite("fall_depth", &EnvConfig::fall_depth)
      .def_readwrite("down_height", &EnvConfig::down_height)
      .def_readwrite("lateral_margin", &EnvConfig::lateral_margin)
      .def_readwrite("stall_time", &EnvConfig::stall_time)
      .def_readwrite("time_limit", &EnvConfig::time_limit)
      .def_readwrite("w_progress", &EnvConfig::w_progress)
      .def_readwrite("w_alive", &EnvConfig::w_alive)
      .def_readwrite("w_time", &EnvConfig::w_time)
      .def_readwrite("w_action", &EnvConfig::w_action)
      .def_readwrite("finish_bonus", &EnvConfig::finish_bonus)
      .def_readwrite("finish_time_bonus", &EnvConfig::finish_time_bonus)
      .def_readwrite("fail_penalty", &EnvConfig::fail_penalty);

  py::class_<VecEnv>(m, "VecEnv")
      .def(py::init([](const EnvConfig& cfg, int num_envs, int threads) {
             py::gil_scoped_release release;
             return std::make_unique<VecEnv>(cfg, num_envs, threads);
           }),
           py::arg("config"), py::arg("num_envs"), py::arg("threads"))
      .def_property_readonly("num_envs", &VecEnv::num_envs)
      .def_property_readonly("obs_size", &VecEnv::obs_size)
      .def_property_readonly("act_size", &VecEnv::act_size)
      .def_property_readonly("threads", &VecEnv::threads)
      .def_property_readonly(
          "obs_segments",  // sizes of the observation parts: body, rays, next obstacles, route
          [](VecEnv& v) {
            const Perception& p = v.env(0).perception();
            const int body = v.obs_size() - p.size();
            return std::vector<int>{body, p.rays_size(), p.lookahead_size(), p.route_size()};
          })
      .def("reset",
           [](VecEnv& v) {
             FloatArray obs({v.num_envs(), v.obs_size()});
             float* o = obs.mutable_data();
             py::gil_scoped_release release;
             v.reset(o);
             return obs;
           })
      .def(
          "step",
          [](VecEnv& v, FloatArray actions) {
            const int n = v.num_envs();
            if (actions.ndim() != 2 || actions.shape(0) != n || actions.shape(1) != v.act_size()) {
              throw std::invalid_argument("actions must have shape (num_envs, act_size)");
            }
            FloatArray obs({n, v.obs_size()}), final_obs({n, v.obs_size()});
            FloatArray reward(n);
            py::array_t<bool> done(n), truncated(n);
            const float* a = actions.data();
            float *o = obs.mutable_data(), *fo = final_obs.mutable_data(), *r = reward.mutable_data();
            auto* d = reinterpret_cast<uint8_t*>(done.mutable_data());
            auto* t = reinterpret_cast<uint8_t*>(truncated.mutable_data());
            {
              py::gil_scoped_release release;
              v.step(a, o, r, d, t, fo);
            }
            return py::make_tuple(obs, reward, done, truncated, final_obs);
          },
          py::arg("actions"),
          "Returns (obs, reward, done, truncated, final_obs); done envs restart automatically.")
      .def("episodes", [](VecEnv& v) {
        py::list out;
        for (const EpisodeStats& e : v.take_episodes()) out.append(episode_dict(e));
        return out;
      });

  py::class_<Policy>(m, "Policy")
      .def(py::init<const std::string&>())
      .def_property_readonly("obs_size", &Policy::obs_size)
      .def_property_readonly("act_size", &Policy::act_size)
      .def_property_readonly("meta", &Policy::meta)
      .def("act", [](const Policy& p, FloatArray obs) {
        std::vector<float> x(obs.data(), obs.data() + obs.size());
        py::array_t<double> out(p.act_size());
        p.act(x, out.mutable_data());
        return out;
      });

  py::list reasons;
  for (int r = 0; r < kNumEndReasons; ++r) reasons.append(end_reason_name(EndReason(r)));
  m.attr("end_reasons") = reasons;
}
