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
#include "core/parkour_env.h"
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

void test_course_generation() {
  std::puts("course generator");
  int counts[kNumObstacleTypes] = {};
  double worst_gap = 0, worst_proj = 0, max_level = 0, max_len = 0;
  bool levels_ok = true, heading_ok = true, sizes_ok = true;
  const int n = 300;
  int same_as_previous = 0;
  std::vector<Obstacle> previous;
  for (int seed = 1; seed <= n; ++seed) {
    CourseParams p;
    p.seed = seed;
    p.difficulty = (seed % 3) * 0.5;
    const Course c = generate_course(p);
    std::vector<Obstacle> types;
    for (const Section& s : c.sections) types.push_back(s.type);
    same_as_previous += types == previous;
    previous = types;
    max_len = std::max(max_len, c.length);
    for (size_t i = 0; i < c.sections.size(); ++i) {
      const Section& s = c.sections[i];
      ++counts[int(s.type)];
      if (i + 1 < c.sections.size()) {  // sections join up, in space and in route distance
        const Section& next = c.sections[i + 1];
        worst_gap = std::max(worst_gap, (s.to_world(s.length, 0, s.level_out) - next.origin).norm() +
                                            std::fabs(s.s0 + s.length - next.s0));
      }
      levels_ok &= s.level_min >= -1e-9 && s.level_min <= std::min(s.level_in, s.level_out) &&
                   std::max(s.level_in, s.level_out) <= 2.4 + 1e-9;
      max_level = std::max(max_level, s.level_out);
      heading_ok &= std::fabs(s.yaw) <= 75 * M_PI / 180 + 1e-9;
      const RoutePoint rp = c.project(s.to_world(s.length / 2, 0.3, 0));
      worst_proj = std::max(worst_proj, std::fabs(rp.s - (s.s0 + s.length / 2)) +
                                            std::fabs(rp.lateral - 0.3) + (rp.section != int(i)));
    }
    for (const Piece& pc : c.pieces) {
      sizes_ok &= pc.size.x > 0 && pc.size.y > 0 && (pc.shape != mjGEOM_BOX || pc.size.z > 0);
    }
  }
  std::printf("  %d courses: ", n);
  for (int t = 1; t < kNumObstacleTypes; ++t) std::printf("%s %d  ", obstacle_name(Obstacle(t)), counts[t]);
  std::printf("\n  longest %.1f m, highest floor %.2f m\n", max_len, max_level);
  bool all_types = true;
  for (int t = 1; t < kNumObstacleTypes; ++t) all_types &= counts[t] > n / 2;
  check(all_types, "every obstacle type is generated");
  check(worst_gap < 1e-9, "sections join up", fmt("(worst %.1e)", worst_gap));
  check(levels_ok, "floor levels within [0, 2.4] m");
  check(heading_ok, "route heading within +-75 deg");
  check(sizes_ok, "all pieces have positive size");
  check(worst_proj < 1e-9, "route projection", fmt("(worst %.1e)", worst_proj));
  check(same_as_previous < 3, "neighbouring seeds give unrelated courses",
        fmt("(%.0f identical obstacle sequences)", same_as_previous));

  CourseParams p;
  p.seed = 42;
  const Course a = generate_course(p), b = generate_course(p);
  bool same = a.pieces.size() == b.pieces.size();
  for (size_t i = 0; same && i < a.pieces.size(); ++i) {
    same = (a.pieces[i].pos - b.pieces[i].pos).norm() == 0 && (a.pieces[i].size - b.pieces[i].size).norm() == 0;
  }
  p.seed = 43;
  check(same && generate_course(p).length != a.length,
        "same seed -> same course, other seed -> other course");
}

// Runs the hold-pose controller inside the env until done or `seconds`; returns the reason.
EndReason run_env_hold(ParkourEnv& env, double seconds) {
  mjData* d = env.sim().data();
  while (d->time < seconds) {
    hold_pose_ctrl(env.sim().model(), d, 0);
    env.sim().substep();
    const StepResult r = env.evaluate();
    if (r.done) return r.reason;
  }
  return EndReason::kNone;
}

// Course with at least one section of the given type.
CourseParams course_with(Obstacle type, uint64_t seed = 1) {
  CourseParams p;
  for (;; ++seed) {
    p.seed = seed;
    for (const Section& s : generate_course(p).sections) {
      if (s.type == type) return p;
    }
  }
}

const Section& first_section(const Course& c, Obstacle type) {
  for (const Section& s : c.sections) {
    if (s.type == type) return s;
  }
  return c.sections.front();
}

void test_course_physics() {
  std::puts("course physics");
  EnvConfig cfg;
  cfg.episodes_per_course = 1000000;
  ParkourEnv env(cfg);
  const mjModel* m = env.sim().model();
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < 5; ++k) env.new_course(env.course().params);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
  m = env.sim().model();
  std::printf("  %zu pieces, %d geoms, course build %.1f ms\n", env.course().pieces.size(), int(m->ngeom), ms);
  bool tagged = true;
  for (size_t i = 0; i < env.course().pieces.size(); ++i) {
    tagged &= std::string(mj_id2name(m, mjOBJ_GEOM, env.first_course_geom() + int(i))) ==
              "course_" + std::to_string(i);
  }
  check(tagged, "course piece i is geom first_course_geom + i");

  env.reset();
  EndReason r = run_env_hold(env, 0.5);
  const double com_z = env.sim().body_state().com.z;
  check(r == EndReason::kNone && com_z > 0.85 && com_z < 1.05, "standing on the start runway",
        std::string("(") + end_reason_name(r) + fmt(", com %.2f m)", com_z));

  // land on a platform pillar: it must carry the body
  env.new_course(course_with(Obstacle::kPlatform));
  const Course& c = env.course();
  const Piece* pillar = nullptr;
  for (const Piece& pc : c.pieces) {
    if (pc.type == Obstacle::kPlatform && pc.size.x > 0.45 && pc.size.y > 0.45) {
      pillar = &pc;
      break;
    }
  }
  const double top = pillar->pos.z + pillar->size.z;
  env.reset();
  env.sim().reset_at({pillar->pos.x, pillar->pos.y, top}, c.sections[pillar->section].yaw, 0.05);
  r = run_env_hold(env, 0.4);
  const double above = env.sim().body_state().com.z - top;
  check(r == EndReason::kNone && above > 0.8, "standing on a platform pillar",
        fmt("(com %.2f m above its top)", above));

  // step into the pit beside a balance beam: must end as a fall
  env.new_course(course_with(Obstacle::kBeam));
  const Section& beam_sec = first_section(env.course(), Obstacle::kBeam);
  Vec3 beam;
  for (const Piece& pc : env.course().pieces) {
    if (pc.type == Obstacle::kBeam) beam = pc.pos;
  }
  const double u = 0.5 * (beam_sec.obstacle_start + beam_sec.obstacle_end);
  Vec3 spot = beam_sec.to_world(u, 0.9, beam_sec.level_in);
  const Vec3 other = beam_sec.to_world(u, -0.9, beam_sec.level_in);
  if (std::hypot(other.x - beam.x, other.y - beam.y) > std::hypot(spot.x - beam.x, spot.y - beam.y)) spot = other;
  env.reset();
  env.sim().reset_at(spot, beam_sec.yaw);
  r = run_env_hold(env, 3.0);
  check(r == EndReason::kFell, "stepping into a pit ends the run", std::string("(") + end_reason_name(r) + ")");

  // standing past the finish line
  env.reset(0.0, env.course().finish_s + 0.5);
  r = env.evaluate().reason;
  check(r == EndReason::kFinished && env.score() > env.course().finish_s, "finish line",
        fmt("(score %.1f)", env.score()));
}

void benchmark(int threads) {
  std::puts("benchmark (random actions on generated courses, new course every 8 episodes)");
  auto run = [&](int nthreads) {
    const int steps = 2000;  // control steps per thread
    std::vector<std::thread> pool;
    std::vector<int> episodes(nthreads);
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < nthreads; ++k) {
      pool.emplace_back([&, k] {
        EnvConfig cfg;
        cfg.course.seed = 1000 * k + 1;
        ParkourEnv env(cfg);
        std::mt19937 rng(k);
        std::uniform_real_distribution<double> u(-0.3, 0.3);
        std::vector<double> act(env.sim().model()->nu);
        std::vector<float> obs;
        for (int t = 0; t < steps; ++t) {
          for (double& x : act) x = u(rng);
          if (env.step(act.data()).done) env.reset(), ++episodes[k];
          env.sim().body_state().flatten(obs, 686.7);
        }
      });
    }
    for (auto& th : pool) th.join();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double sps = nthreads * steps / sec;
    int eps = 0;
    for (int e : episodes) eps += e;
    std::printf("  %d thread(s): %8.0f control steps/s  (%5.1fx realtime, %.1f M steps/hour, %d episodes)\n",
                nthreads, sps, sps * 0.02, sps * 3600 / 1e6, eps);
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
  test_course_generation();
  test_course_physics();
  benchmark(threads);
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
