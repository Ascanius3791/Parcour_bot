// Physics sanity checks and throughput benchmark for the humanoid.
// Usage: parcour_selftest [scene.xml] [--threads N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/controllers.h"
#include "core/simulation.h"

using namespace parcour;

namespace {
int failures = 0;

void check(bool ok, const char* what, const std::string& detail = "") {
  std::printf("  [%s] %s %s\n", ok ? " ok " : "FAIL", what, detail.c_str());
  if (!ok) ++failures;
}

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b, c);
  return buf;
}

void test_model(Simulation& sim) {
  std::puts("model");
  const mjModel* m = sim.model();
  const auto& L = sim.layout();
  check(std::fabs(L.total_mass - 70.0) < 0.1, "total mass 70 kg", fmt("(%.2f kg)", L.total_mass));

  sim.reset();
  mjData* d = sim.data();
  // top of head = head sphere top
  const int head_geom = mj_name2id(m, mjOBJ_GEOM, "head_1");
  const double top = d->geom_xpos[3 * head_geom + 2] + m->geom_size[3 * head_geom];
  check(std::fabs(top - 1.75) < 0.03, "standing height 1.75 m", fmt("(%.3f m)", top));
  std::printf("  dofs %d, actuators %d, actuated joints %zu, observation size %d\n", int(m->nv),
              int(m->nu), L.actuated_joints.size(),
              BodyState::flat_size(int(L.actuated_joints.size())));

  // no self-intersection in the default pose: only shoe/floor contacts allowed after settling
  sim.reset(0.0);
  int bad = 0;
  for (int i = 0; i < d->ncon; ++i) {
    const int b1 = m->geom_bodyid[d->contact[i].geom[0]], b2 = m->geom_bodyid[d->contact[i].geom[1]];
    if (b1 != 0 && b2 != 0) {
      ++bad;
      std::printf("    self contact: %s <-> %s\n", mj_id2name(m, mjOBJ_BODY, b1),
                  mj_id2name(m, mjOBJ_BODY, b2));
    }
  }
  check(bad == 0, "no self-contacts in default pose");
}

void test_split_step_matches_mj_step(Simulation& sim) {
  std::puts("integration");
  const mjModel* m = sim.model();
  mjData* a = mj_makeData(m);
  mjData* b = mj_makeData(m);
  mj_forward(m, a);
  mj_forward(m, b);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-0.3, 0.3);
  for (int t = 0; t < 300; ++t) {
    for (int i = 0; i < m->nu; ++i) a->ctrl[i] = b->ctrl[i] = u(rng) * m->actuator_ctrlrange[2 * i + 1];
    mj_step(m, a);
    mj_step2(m, b);
    mj_step1(m, b);
  }
  double diff = 0;
  for (int i = 0; i < m->nq; ++i) diff = std::max(diff, std::fabs(a->qpos[i] - b->qpos[i]));
  check(diff < 1e-9, "mj_step2/mj_step1 loop equals mj_step", fmt("(max |dq| %.2e)", diff));

  // the joint reaction forces must be computed inside mj_step2 (triggered by the IMU sensor):
  // compare with a full mj_forward + mj_rnePostConstraint of the same pre-step state
  for (int i = 0; i < m->nu; ++i) b->ctrl[i] = 0;
  mjData* c = mj_copyData(nullptr, m, b);
  mj_forward(m, c);
  mj_rnePostConstraint(m, c);
  mj_step2(m, b);
  double d2 = 0, mag = 0;
  for (int i = 0; i < 6 * m->nbody; ++i) {
    d2 = std::max(d2, std::fabs(c->cfrc_int[i] - b->cfrc_int[i]));
    mag = std::max(mag, std::fabs(c->cfrc_int[i]));
  }
  mj_deleteData(c);
  check(mag > 1 && d2 < 1e-3 * mag, "joint reaction forces available after mj_step2",
        fmt("(max |F| %.1f, diff %.2e)", mag, d2));
  mj_deleteData(a);
  mj_deleteData(b);
}

void test_momentum_bookkeeping(Simulation& sim) {
  std::puts("sensors");
  const mjModel* m = sim.model();
  mjData* d = sim.data();
  sim.reset(0.5);
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> act(m->nu);
  for (int t = 0; t < 20; ++t) {
    for (double& x : act) x = u(rng);
    sim.step(act.data());
  }
  const BodyState& s = sim.body_state();
  // sum of per-body angular momentum about the com must equal the total
  Vec3 sum;
  const Vec3 com = s.com;
  for (int b : sim.layout().bodies) {
    double v6[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, b, v6, 0);
    const Vec3 w = Vec3::from(v6), v = Vec3::from(v6 + 3);
    const double* R = d->ximat + 9 * b;
    const double* I = m->body_inertia + 3 * b;
    Vec3 Iw;
    for (int i = 0; i < 3; ++i) {
      const double li = I[i] * (R[i] * w.x + R[3 + i] * w.y + R[6 + i] * w.z);
      Iw += Vec3{R[i], R[3 + i], R[6 + i]} * li;
    }
    sum += Iw + (Vec3::from(d->xipos + 3 * b) - com).cross(v * m->body_mass[b]);
  }
  const double err = (sum - s.ang_momentum).norm();
  check(err < 1e-6 * std::max(1.0, s.ang_momentum.norm()), "angular momentum bookkeeping",
        fmt("(|L| %.3f, err %.1e)", s.ang_momentum.norm(), err));
  std::vector<float> flat;
  s.flatten(flat, sim.layout().total_mass * 9.81);
  check(int(flat.size()) == BodyState::flat_size(int(s.joint_pos.size())), "observation size",
        fmt("(%.0f)", double(flat.size())));
}

// Runs the hold-pose controller; returns the broken segment (-1 if none).
int run_hold(Simulation& sim, double lift, double seconds) {
  sim.reset(lift);
  mjData* d = sim.data();
  while (d->time < seconds) {
    hold_pose_ctrl(sim.model(), d, 0);
    if (!sim.substep()) return sim.injury().broken_segment();
  }
  return -1;
}

// Runs without any muscle torque; returns the broken segment (-1 if none). Optionally
// averages the left knee force over the window [0.1 s, 0.3 s], while the body still stands.
int run_limp(Simulation& sim, double lift, double seconds, double* knee_force = nullptr) {
  sim.reset(lift);
  mjData* d = sim.data();
  const int knee = 11;  // shank_l
  double fsum = 0;
  int n = 0;
  while (d->time < seconds) {
    mju_zero(d->ctrl, sim.model()->nu);
    if (!sim.substep()) return sim.injury().broken_segment();
    if (d->time > 0.1 && d->time < 0.3) {
      fsum += sim.injury().loads()[knee].force;
      if (knee_force) *knee_force = fsum / ++n;
    }
  }
  return -1;
}

void test_injuries(Simulation& sim) {
  std::puts("injuries");
  double knee_force = 0;
  // a stiff pose controller cannot balance, so only check the first half second
  int broke = run_hold(sim, 0.0, 0.5);
  const auto& L = sim.layout();
  const double expected =
      0.5 * 9.81 * (L.total_mass - 2 * (sim.model()->body_mass[L.segments[11]] +
                                        sim.model()->body_mass[L.segments[12]] +
                                        sim.model()->body_mass[L.segments[13]]));
  check(broke < 0, "standing with pose controller: intact");
  broke = run_limp(sim, 0.0, 3.0, &knee_force);
  std::printf("  [info] collapse from standing, limp: %s\n",
              broke >= 0 ? (InjuryMonitor::joint_name(broke) + " broke").c_str() : "intact");
  check(std::fabs(knee_force - expected) < 0.1 * expected, "standing knee load = half upper body weight",
        fmt("(%.0f N, expected %.0f N)", knee_force, expected));
  broke = run_hold(sim, 0.3, 1.5);
  check(broke < 0, "30 cm drop, braced: intact",
        broke >= 0 ? "(" + InjuryMonitor::joint_name(broke) + ")" : "");
  broke = run_limp(sim, 6.0, 4.0);
  check(broke >= 0, "6 m fall, limp: something breaks",
        broke >= 0 ? "(" + InjuryMonitor::joint_name(broke) + ")" : "");
}

void test_joint_limits(Simulation& sim) {
  std::puts("joint limits");
  const mjModel* m = sim.model();
  mjData* d = sim.data();
  sim.reset(1.0);
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> act(m->nu);
  double worst = 0;
  int worst_j = -1;
  for (int t = 0; t < 200; ++t) {
    for (double& x : act) x = u(rng) > 0 ? 1 : -1;  // bang-bang at full strength
    sim.step(act.data());
    for (int j : sim.layout().actuated_joints) {
      const double q = d->qpos[m->jnt_qposadr[j]];
      const double v = std::max(m->jnt_range[2 * j] - q, q - m->jnt_range[2 * j + 1]);
      if (v > worst) worst = v, worst_j = j;
    }
  }
  check(worst < 0.1, "max-torque flailing stays within joint ranges (+-6 deg)",
        fmt("(worst %.1f deg)", worst * 180 / M_PI) +
            (worst_j >= 0 ? std::string(" at ") + mj_id2name(m, mjOBJ_JOINT, worst_j) : ""));
}

void benchmark(const std::string& path, int threads) {
  std::puts("benchmark (random actions, sensors + injury monitor every step)");
  auto run = [&](int nthreads) {
    const int steps = 2000;  // control steps per thread
    std::vector<std::thread> pool;
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < nthreads; ++k) {
      pool.emplace_back([&, k] {
        Simulation sim(path);
        std::mt19937 rng(k);
        std::uniform_real_distribution<double> u(-0.3, 0.3);
        std::vector<double> act(sim.model()->nu);
        std::vector<float> obs;
        for (int t = 0; t < steps; ++t) {
          for (double& x : act) x = u(rng);
          if (!sim.step(act.data()) || t % 100 == 99) sim.reset();
          sim.body_state().flatten(obs, 686.7);
        }
      });
    }
    for (auto& th : pool) th.join();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double sps = nthreads * steps / sec;
    std::printf("  %d thread(s): %8.0f control steps/s  (%5.1fx realtime, %.1f M steps/hour)\n",
                nthreads, sps, sps * 0.02, sps * 3600 / 1e6);
  };
  run(1);
  if (threads > 1) run(threads);
}
}  // namespace

int main(int argc, char** argv) {
  std::string path = std::string(PARCOUR_ASSET_DIR) + "/scene_flat.xml";
  int threads = int(std::thread::hardware_concurrency());
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    else path = argv[i];
  }
  Simulation sim(path);
  std::printf("MuJoCo %s, scene %s\n", mj_versionString(), path.c_str());
  test_model(sim);
  test_split_step_matches_mj_step(sim);
  test_momentum_bookkeeping(sim);
  test_injuries(sim);
  test_joint_limits(sim);
  benchmark(path, threads);
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
