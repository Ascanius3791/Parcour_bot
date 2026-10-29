// Interactive 3D viewer for the parkour humanoid.
// Usage: parcour_viewer [--policy policy.bin] [--seed N] [--difficulty D] [--obstacles N]
//                       [--track LENGTH] [--section K] [--overview] [--vision]
//                       [--screenshot out.ppm --at SECONDS]
//        parcour_viewer scene.xml        (a fixed scene instead of generated courses)
//
// Shows the body, its centre of mass, momentum and angular momentum, and colours every
// segment by the stress on the joint above it (skin -> red; magenta = broken). On generated
// courses it also shows the route, progress, how the run ended and what the bot perceives
// (rays coloured by obstacle type, upcoming obstacles). With --policy it runs a trained policy
// (from train/) on the course it was trained for, unless --obstacles / --track / --difficulty
// say otherwise. Press H for the key list.

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/controllers.h"
#include "core/parkour_env.h"
#include "core/policy.h"
#include "core/simulation.h"

using namespace parcour;

namespace {

enum class Mode { kLimp, kHoldPose, kPolicy };

struct App {
  Simulation* sim = nullptr;
  std::unique_ptr<Simulation> fixed_scene;  // when a scene file is given
  std::unique_ptr<ParkourEnv> env;          // generated courses otherwise
  CourseParams course;
  double start_s = -1;
  mjvCamera cam;
  mjvOption opt;
  mjvScene scn;
  mjrContext con;
  mjvPerturb pert;

  Mode mode = Mode::kHoldPose;
  bool paused = false, step_once = false;
  bool grip = false, invincible = false;
  bool show_com = true, show_limbs = false, show_stress = true, show_help = false;
  bool tracking = true, show_route = true, show_vision = false;
  std::vector<float> obs;  // policy input, recomputed every frame for the vision overlay
  std::unique_ptr<Policy> policy;
  std::vector<float> policy_obs;
  std::vector<double> action;
  long substeps = 0;  // since reset; the policy acts every frame_skip substeps
  double drop_height = 2.0;

  // mouse state
  bool btn_left = false, btn_right = false;
  double lastx = 0, lasty = 0, last_click = 0;

  // real-time bookkeeping
  double sim_wall_offset = 0;
  double last_wall = 0;
  double rtf = 0;
};

App app;

const char* kHelp =
    "Space\nRight\nBackspace\nD\nM\nG\nI\nC\nL\nS\nT\n+ / -\n"
    "N / P\n[ / ]\nJ / K\nR\nY\nV\n"
    "Dbl-click\nCtrl+L drag\nCtrl+R drag\nH";
const char* kHelpText =
    "pause / run\nsingle step (paused)\nreset at start\ndrop from height\n"
    "mode: limp / hold pose / policy\ngrip on / off\ninvincible (no breaking)\ncom + momentum\n"
    "per-limb momentum\nstress colours\ntrack camera\ndrop height\n"
    "next / previous course\ndifficulty - / +\nstart at next / previous obstacle\n"
    "route + progress\nvision: rays + next obstacles\ncourse overview camera\n"
    "select body\nrotate body\npull body\nhelp";

void reset(double lift) {
  if (app.env) app.env->reset(lift, app.start_s);
  else app.sim->reset(lift);
  app.substeps = 0;
  app.sim_wall_offset = glfwGetTime() - app.sim->data()->time;
  mjv_defaultPerturb(&app.pert);
  app.pert.select = 0;
}

// Builds the course in app.course and rebuilds everything that depends on the model.
void load_course() {
  app.env->new_course(app.course);
  const mjModel* m = app.sim->model();
  mjv_freeScene(&app.scn);
  mjv_makeScene(m, &app.scn, 4000);
  mjr_freeContext(&app.con);
  mjr_makeContext(m, &app.con, mjFONTSCALE_150);
  app.start_s = -1;
  reset(0.0);
  app.paused = false;
}

// Free camera looking at the whole course from the side.
void overview_camera() {
  if (!app.env) return;
  const Course& c = app.env->course();
  Vec3 lo{1e9, 1e9, 1e9}, hi{-1e9, -1e9, -1e9};
  for (const Section& s : c.sections) {
    for (double u : {0.0, s.length}) {
      const Vec3 p = s.to_world(u, 0, u > 0 ? s.level_out : s.level_in);
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
  }
  app.tracking = false;
  app.cam.type = mjCAMERA_FREE;
  ((lo + hi) * 0.5).to(app.cam.lookat);
  app.cam.distance = 0.85 * std::max(hi.x - lo.x, hi.y - lo.y) + 4;
  app.cam.azimuth = -60;
  app.cam.elevation = -30;
}

// Moves the start to the approach of the next (+1) or previous (-1) obstacle.
void jump_section(int dir) {
  if (!app.env) return;
  const Course& c = app.env->course();
  const int n = int(c.sections.size());
  const int cur = app.start_s < 0 ? 0 : c.section_at(app.start_s);
  const int next = std::clamp(cur + dir, 0, n - 1);
  app.start_s = next == 0 ? -1 : c.sections[next].s0 + 0.5;
  reset(0.0);
  app.paused = false;
}

void apply_controller() {
  const mjModel* m = app.sim->model();
  mjData* d = app.sim->data();
  if (app.mode == Mode::kPolicy) {
    // same timing as training: observe and act at the control rate, PD on every substep
    if (app.substeps % app.sim->frame_skip == 0) {
      app.env->observe(app.policy_obs);
      app.policy->act(app.policy_obs, app.action.data());
    }
    app.sim->layout().apply_pd_action(m, d, app.action.data());
  } else if (app.mode == Mode::kHoldPose) {
    hold_pose_ctrl(m, d, app.grip ? 1.0 : 0.0);
  } else {
    mju_zero(d->ctrl, m->nu);
    for (int a = 0; a < m->nu; ++a) {
      if (m->actuator_trntype[a] == mjTRN_TENDON && app.grip) {
        d->ctrl[a] = m->actuator_ctrlrange[2 * a + 1];
      }
    }
  }
}

void physics() {
  const mjModel* m = app.sim->model();
  mjData* d = app.sim->data();
  const double wall = glfwGetTime();
  if (app.paused && !app.step_once) {
    app.sim_wall_offset = wall - d->time;
    app.last_wall = wall;
    app.rtf = 0;
    return;
  }
  const double t0 = d->time;
  const double target = app.step_once ? d->time + m->opt.timestep : wall - app.sim_wall_offset;
  const double budget_end = wall + 1.0 / 30;  // keep the UI responsive if we fall behind
  while (d->time < target && glfwGetTime() < budget_end) {
    mju_zero(d->xfrc_applied, 6 * m->nbody);
    mjv_applyPerturbPose(m, d, &app.pert, 0);
    mjv_applyPerturbForce(m, d, &app.pert);
    apply_controller();
    const bool ok = app.sim->substep();
    ++app.substeps;
    if (!ok && !app.invincible) {
      app.paused = true;
      break;
    }
  }
  if (app.env) app.env->evaluate();
  if (d->time < target && !app.step_once) app.sim_wall_offset = wall - d->time;  // drop lag
  app.step_once = false;
  const double dt_wall = std::max(wall - app.last_wall, 1e-3);
  app.last_wall = wall;
  app.rtf = 0.9 * app.rtf + 0.1 * (d->time - t0) / dt_wall;
}

// ---------------------------------------------------------------- overlays

void add_arrow(const Vec3& from, const Vec3& vec, double width, const float rgba[4]) {
  if (app.scn.ngeom >= app.scn.maxgeom || vec.norm() < 1e-3) return;
  mjvGeom* g = app.scn.geoms + app.scn.ngeom++;
  mjv_initGeom(g, mjGEOM_ARROW, nullptr, nullptr, nullptr, rgba);
  double a[3], b[3];
  from.to(a);
  (from + vec).to(b);
  mjv_connector(g, mjGEOM_ARROW, width, a, b);
  g->category = mjCAT_DECOR;
}

void add_sphere(const Vec3& p, double r, const float rgba[4]) {
  if (app.scn.ngeom >= app.scn.maxgeom) return;
  mjvGeom* g = app.scn.geoms + app.scn.ngeom++;
  const double size[3] = {r, r, r};
  double pos[3];
  p.to(pos);
  mjv_initGeom(g, mjGEOM_SPHERE, size, pos, nullptr, rgba);
  g->category = mjCAT_DECOR;
}

void add_line(const Vec3& from, const Vec3& to, const float rgba[4]) {
  if (app.scn.ngeom >= app.scn.maxgeom) return;
  mjvGeom* g = app.scn.geoms + app.scn.ngeom++;
  mjv_initGeom(g, mjGEOM_LINE, nullptr, nullptr, nullptr, rgba);
  double a[3], b[3];
  from.to(a);
  to.to(b);
  mjv_connector(g, mjGEOM_LINE, 2, a, b);
  g->category = mjCAT_DECOR;
}

void add_box(const Vec3& centre, const Vec3& half, double yaw, const float rgba[4]) {
  if (app.scn.ngeom >= app.scn.maxgeom) return;
  mjvGeom* g = app.scn.geoms + app.scn.ngeom++;
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double mat[9] = {c, -s, 0, s, c, 0, 0, 0, 1};
  double size[3], pos[3];
  half.to(size);
  centre.to(pos);
  mjv_initGeom(g, mjGEOM_BOX, size, pos, mat, rgba);
  g->category = mjCAT_DECOR;
}

// bright colour per obstacle type for the vision overlay
const float kTypeColour[kNumObstacleTypes][4] = {
    {0.85, 0.85, 0.85, 1}, {1.0, 0.9, 0.1, 1}, {1.0, 0.5, 0.0, 1}, {1.0, 0.15, 0.15, 1},
    {0.9, 0.2, 0.9, 1},    {0.1, 0.9, 0.9, 1}, {0.2, 0.45, 1.0, 1}};

void decorate_vision() {
  app.env->observe(app.obs);  // casts the rays
  const ParkourEnv& e = *app.env;
  const Perception& p = e.perception();
  for (const RayHit& h : p.hits()) {
    if (h.piece < 0) continue;  // nothing within range
    const Vec3 hit = p.eye() + h.dir * h.dist;
    float c[4];
    std::copy(kTypeColour[int(h.type)], kTypeColour[int(h.type)] + 4, c);
    c[3] = 0.5f;
    add_line(p.eye(), hit, c);
    add_sphere(hit, 0.025, kTypeColour[int(h.type)]);
  }
  // key pieces of the upcoming obstacles: the next one brightest
  const Course& c = e.course();
  const std::vector<int> next = upcoming_obstacles(c, e.progress(), p.config().lookahead);
  for (size_t k = 0; k < next.size(); ++k) {
    const Section& sec = c.sections[next[k]];
    float col[4];
    std::copy(kTypeColour[int(sec.type)], kTypeColour[int(sec.type)] + 4, col);
    col[3] = k == 0 ? 0.45f : 0.2f;
    for (int i : key_pieces(c, next[k], p.config().key_pieces)) {
      const Piece& pc = c.pieces[i];
      // grow the box a little so it shows around the piece; clip pillars at 1 m deep
      const Vec3 lo{pc.lo.x - 0.02, pc.lo.y - 0.02, std::max(pc.lo.z, pc.hi.z - 1.0)};
      const Vec3 hi{pc.hi.x + 0.02, pc.hi.y + 0.02, pc.hi.z + 0.02};
      const Vec3 mid = (lo + hi) * 0.5;
      add_box(sec.to_world(mid.x, mid.y, mid.z), (hi - lo) * 0.5, sec.yaw, col);
    }
  }
}

void decorate() {
  const mjModel* m = app.sim->model();
  const auto& layout = app.sim->layout();
  const BodyState& s = app.sim->body_state();

  if (app.show_stress) {
    const int broken = app.sim->injury().broken_segment();
    for (int i = 0; i < app.scn.ngeom; ++i) {
      mjvGeom* g = app.scn.geoms + i;
      if (g->objtype != mjOBJ_GEOM) continue;
      int b = m->geom_bodyid[g->objid];
      int seg = layout.segment_of_body(b);
      while (seg < 0 && b > 0) seg = layout.segment_of_body(b = m->body_parentid[b]);  // fingers
      if (seg < 0) continue;
      if (seg == broken) {
        g->rgba[0] = 1, g->rgba[1] = 0, g->rgba[2] = 1;
        continue;
      }
      const float t = float(std::clamp(s.stress[seg], 0.0, 1.0));
      g->rgba[0] = g->rgba[0] * (1 - t) + 1.0f * t;
      g->rgba[1] = g->rgba[1] * (1 - t) + 0.1f * t;
      g->rgba[2] = g->rgba[2] * (1 - t) + 0.1f * t;
    }
  }

  if (app.env && app.show_route) {
    const Course& c = app.env->course();
    const float white[4] = {1, 1, 1, 0.8}, green[4] = {0.2, 1, 0.3, 1}, gold[4] = {1, 0.8, 0.1, 1};
    for (const Section& sec : c.sections) {
      // centreline over the flat parts (approach and runout) of every section
      add_line(sec.to_world(0, 0, sec.level_in + 0.01),
               sec.to_world(sec.obstacle_start, 0, sec.level_in + 0.01), white);
      add_line(sec.to_world(sec.obstacle_end, 0, sec.level_out + 0.01),
               sec.to_world(sec.length, 0, sec.level_out + 0.01), white);
      // checkpoint at the end of every section, green once reached
      const bool reached = app.env->best_progress() >= sec.s0 + sec.length;
      const Vec3 cp = sec.to_world(sec.length, 0, sec.level_out);
      add_line(cp, cp + Vec3{0, 0, 0.6}, reached ? green : white);
      add_sphere(cp + Vec3{0, 0, 0.6}, 0.04, reached ? green : white);
    }
    const Section& last = c.sections.back();
    const double w = c.params.width / 2, uf = c.finish_s - last.s0;
    add_line(last.to_world(uf, -w, last.level_out + 0.01), last.to_world(uf, w, last.level_out + 0.01),
             gold);
    // best progress so far
    const double best = app.env->best_progress();
    const Section& bs = c.sections[c.section_at(best)];
    const Vec3 bp = bs.to_world(best - bs.s0, 0, std::max(bs.level_in, bs.level_out) + 0.03);
    add_sphere(bp, 0.05, gold);
  }

  if (app.env && app.show_vision) decorate_vision();

  if (app.show_com) {
    const float red[4] = {1, 0.2, 0.2, 1}, yellow[4] = {1, 0.85, 0.1, 1},
                blue[4] = {0.2, 0.5, 1, 1}, grey[4] = {1, 0.2, 0.2, 0.6};
    add_sphere(s.com, 0.035, red);
    // ground projection onto the floor under the com (z = 0 in a fixed scene)
    double ground = 0;
    if (app.env) {
      const Section& sec = app.env->course().sections[app.env->section()];
      ground = s.com.z > sec.level_out - 0.3 ? sec.level_out : sec.level_in;
    }
    add_line(s.com, {s.com.x, s.com.y, ground}, grey);
    add_sphere({s.com.x, s.com.y, ground + 0.003}, 0.02, grey);
    add_arrow(s.com, s.momentum * 0.005, 0.02, yellow);    // 1 m per 200 kg m/s
    add_arrow(s.com, s.ang_momentum * 0.05, 0.02, blue);   // 1 m per 20 kg m^2/s
  }
  if (app.show_limbs) {
    const float green[4] = {0.2, 0.9, 0.3, 1};
    for (const LimbState& L : s.limbs) add_arrow(s.com + L.pos, L.momentum * 0.02, 0.008, green);
  }
}

std::string mode_name() {
  switch (app.mode) {
    case Mode::kLimp: return "limp";
    case Mode::kHoldPose: return "hold pose";
    case Mode::kPolicy: {
      const auto stage = app.policy->meta().find("stage");
      char buf[96];
      std::snprintf(buf, sizeof(buf), "policy (%s, %.1f M steps)",
                    stage == app.policy->meta().end() ? "?" : stage->second.c_str(),
                    app.policy->meta_num("steps", 0) / 1e6);
      return buf;
    }
  }
  return "?";
}

void overlay(mjrRect vp) {
  const mjData* d = app.sim->data();
  const BodyState& s = app.sim->body_state();
  const auto& inj = app.sim->injury();
  int worst = 1;
  for (int i = 1; i < kNumSegments; ++i) {
    if (inj.loads()[i].stress > inj.loads()[worst].stress) worst = i;
  }
  char left[512], right[512];
  std::snprintf(left, sizeof(left),
                "time\nspeed\nmode\ngrip\ncom height\ncom speed\n|momentum|\n|ang. momentum|\n"
                "max stress");
  std::snprintf(right, sizeof(right),
                "%.2f s%s\n%.2fx real time\n%s%s\n%s\n%.2f m\n%.2f m/s\n%.1f kg m/s\n"
                "%.2f kg m^2/s\n%.0f%% (%s)",
                d->time, app.paused ? "  [paused]" : "", app.rtf,
                mode_name().c_str(), app.invincible ? " (invincible)" : "",
                app.grip ? "closed" : "open", s.com.z, s.com_vel.norm(), s.momentum.norm(),
                s.ang_momentum.norm(), 100 * inj.loads()[worst].stress,
                InjuryMonitor::joint_name(worst).c_str());
  if (app.env) {
    const ParkourEnv& e = *app.env;
    const Course& c = e.course();
    char l2[160], r2[320];
    std::snprintf(l2, sizeof(l2), "\ncourse\nsection\nprogress\nscore");
    std::snprintf(r2, sizeof(r2), "\nseed %llu, difficulty %.1f\n%d / %d  %s\n%.1f / %.1f m\n%.1f",
                  (unsigned long long)c.params.seed, c.params.difficulty, e.section() + 1,
                  int(c.sections.size()), obstacle_name(c.sections[e.section()].type),
                  e.best_progress(), c.finish_s, e.score());
    std::strncat(left, l2, sizeof(left) - std::strlen(left) - 1);
    std::strncat(right, r2, sizeof(right) - std::strlen(right) - 1);
    if (app.show_vision) {
      std::string next;
      for (int i : upcoming_obstacles(c, e.progress(), e.perception().config().lookahead)) {
        const Section& sec = c.sections[i];
        char item[64];
        std::snprintf(item, sizeof(item), "%s%s %.1f m", next.empty() ? "" : ", ",
                      obstacle_name(sec.type), std::max(0.0, sec.s0 + sec.obstacle_start - e.progress()));
        next += item;
      }
      char l3[32], r3[160];
      std::snprintf(l3, sizeof(l3), "\nnext");
      std::snprintf(r3, sizeof(r3), "\n%s", next.empty() ? "finish" : next.c_str());
      std::strncat(left, l3, sizeof(left) - std::strlen(left) - 1);
      std::strncat(right, r3, sizeof(right) - std::strlen(right) - 1);
    }
  }
  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, left, right, &app.con);

  if (inj.broken_segment() >= 0) {
    std::string msg = "JOINT BROKEN: " + InjuryMonitor::joint_name(inj.broken_segment());
    mjr_overlay(mjFONT_BIG, mjGRID_TOP, vp, msg.c_str(),
                app.invincible ? nullptr : "Backspace = reset, I = continue invincible", &app.con);
  } else if (app.env && app.env->end_reason() != EndReason::kNone) {
    std::string msg = end_reason_name(app.env->end_reason());
    for (char& ch : msg) ch = char(std::toupper(ch));
    mjr_overlay(mjFONT_BIG, mjGRID_TOP, vp, msg.c_str(), "Backspace = restart", &app.con);
  }
  if (app.show_help) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPRIGHT, vp, kHelp, kHelpText, &app.con);
  } else {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMRIGHT, vp, "H: help", nullptr, &app.con);
  }
}

// ---------------------------------------------------------------- input

void on_key(GLFWwindow*, int key, int, int act, int) {
  if (act == GLFW_RELEASE) return;
  switch (key) {
    case GLFW_KEY_SPACE: app.paused = !app.paused; break;
    case GLFW_KEY_RIGHT: if (app.paused) app.step_once = true; break;
    case GLFW_KEY_BACKSPACE: reset(0.0); app.paused = false; break;
    case GLFW_KEY_D: reset(app.drop_height); app.paused = false; break;
    case GLFW_KEY_M:
      if (app.mode == Mode::kLimp) {
        app.mode = Mode::kHoldPose;
      } else if (app.mode == Mode::kHoldPose && app.policy) {
        app.mode = Mode::kPolicy;
        reset(0.0);
      } else {
        app.mode = Mode::kLimp;
      }
      break;
    case GLFW_KEY_G: app.grip = !app.grip; break;
    case GLFW_KEY_I:
      app.invincible = !app.invincible;
      if (app.invincible) app.paused = false;
      break;
    case GLFW_KEY_C: app.show_com = !app.show_com; break;
    case GLFW_KEY_L: app.show_limbs = !app.show_limbs; break;
    case GLFW_KEY_S: app.show_stress = !app.show_stress; break;
    case GLFW_KEY_H: app.show_help = !app.show_help; break;
    case GLFW_KEY_T:
      app.tracking = !app.tracking;
      app.cam.type = app.tracking ? mjCAMERA_TRACKING : mjCAMERA_FREE;
      break;
    case GLFW_KEY_EQUAL: case GLFW_KEY_KP_ADD: app.drop_height += 0.5; break;
    case GLFW_KEY_MINUS: case GLFW_KEY_KP_SUBTRACT:
      app.drop_height = std::max(0.0, app.drop_height - 0.5);
      break;
    case GLFW_KEY_N: case GLFW_KEY_P:
      if (app.env) {
        app.course.seed = key == GLFW_KEY_N ? app.course.seed + 1 : std::max<uint64_t>(1, app.course.seed - 1);
        load_course();
      }
      break;
    case GLFW_KEY_LEFT_BRACKET: case GLFW_KEY_RIGHT_BRACKET:
      if (app.env) {
        const double step = key == GLFW_KEY_RIGHT_BRACKET ? 0.1 : -0.1;
        app.course.difficulty = std::clamp(std::round((app.course.difficulty + step) * 10) / 10, 0.0, 1.0);
        load_course();
      }
      break;
    case GLFW_KEY_J: jump_section(+1); break;
    case GLFW_KEY_K: jump_section(-1); break;
    case GLFW_KEY_R: app.show_route = !app.show_route; break;
    case GLFW_KEY_Y: app.show_vision = !app.show_vision; break;
    case GLFW_KEY_V: overview_camera(); break;
    default: break;
  }
}

bool ctrl_down(GLFWwindow* w) {
  return glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
         glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
}

void on_button(GLFWwindow* w, int button, int act, int) {
  app.btn_left = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  app.btn_right = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  glfwGetCursorPos(w, &app.lastx, &app.lasty);
  const mjModel* m = app.sim->model();
  mjData* d = app.sim->data();

  if (act == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
    const double now = glfwGetTime();
    if (now - app.last_click < 0.3) {  // double click: select body under cursor
      int width, height;
      glfwGetWindowSize(w, &width, &height);
      mjtNum selpnt[3];
      int geom = -1;
      const int body = mjv_select(m, d, &app.opt, double(width) / height, app.lastx / width,
                                  (height - app.lasty) / height, &app.scn, selpnt, &geom, nullptr,
                                  nullptr);
      app.pert.select = body > 0 ? body : 0;
      if (body > 0) {
        mjtNum tmp[3];
        mju_sub3(tmp, selpnt, d->xpos + 3 * body);
        mju_mulMatTVec(app.pert.localpos, d->xmat + 9 * body, tmp, 3, 3);
      }
    }
    app.last_click = now;
  }
  // ctrl + drag on a selected body: perturb it
  if (act == GLFW_PRESS && ctrl_down(w) && app.pert.select > 0) {
    app.pert.active = button == GLFW_MOUSE_BUTTON_RIGHT ? mjPERT_TRANSLATE : mjPERT_ROTATE;
    mjv_initPerturb(m, d, &app.scn, &app.pert);
  }
  if (act == GLFW_RELEASE) app.pert.active = 0;
}

void on_move(GLFWwindow* w, double x, double y) {
  if (!app.btn_left && !app.btn_right) return;
  const double dx = x - app.lastx, dy = y - app.lasty;
  app.lastx = x, app.lasty = y;
  int width, height;
  glfwGetWindowSize(w, &width, &height);
  const bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  mjtMouse action;
  if (app.btn_right) action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  if (app.pert.active) {
    mjv_movePerturb(app.sim->model(), app.sim->data(), action, dx / height, dy / height,
                    &app.scn, &app.pert);
  } else {
    mjv_moveCamera(app.sim->model(), action, dx / height, dy / height, &app.cam);
  }
}

void on_scroll(GLFWwindow*, double, double yoff) {
  mjv_moveCamera(app.sim->model(), mjMOUSE_ZOOM, 0, -0.05 * yoff, &app.cam);
}

void save_ppm(const char* file, mjrRect vp) {
  std::vector<unsigned char> rgb(3 * vp.width * vp.height);
  mjr_readPixels(rgb.data(), nullptr, vp, &app.con);
  FILE* f = std::fopen(file, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%d %d\n255\n", vp.width, vp.height);
  for (int row = vp.height - 1; row >= 0; --row) {  // OpenGL rows are bottom-up
    std::fwrite(rgb.data() + 3 * row * vp.width, 1, 3 * vp.width, f);
  }
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  const char* screenshot = nullptr;
  double shot_time = 0;
  bool overview = false;
  int start_section = 0;
  const char* policy_path = nullptr;
  bool course_given = false;
  double track = -1;
  for (int i = 1; i < argc; ++i) {
    course_given |= !std::strcmp(argv[i], "--difficulty") || !std::strcmp(argv[i], "--obstacles") ||
                    !std::strcmp(argv[i], "--track");
    if (!std::strcmp(argv[i], "--screenshot") && i + 1 < argc) screenshot = argv[++i];
    else if (!std::strcmp(argv[i], "--at") && i + 1 < argc) shot_time = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) app.course.seed = std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--difficulty") && i + 1 < argc) app.course.difficulty = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--obstacles") && i + 1 < argc) app.course.num_obstacles = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--section") && i + 1 < argc) start_section = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--track") && i + 1 < argc) track = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--policy") && i + 1 < argc) policy_path = argv[++i];
    else if (!std::strcmp(argv[i], "--overview")) overview = true;
    else if (!std::strcmp(argv[i], "--vision")) app.show_vision = true;
    else path = argv[i];
  }
  if (track >= 0) app.course.num_obstacles = 0, app.course.finish_length = track;
  if (policy_path) {
    try {
      app.policy = std::make_unique<Policy>(policy_path);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
    if (!course_given) {  // the course the policy was trained on
      app.course.num_obstacles = int(app.policy->meta_num("course_num_obstacles", app.course.num_obstacles));
      app.course.finish_length = app.policy->meta_num("course_finish_length", app.course.finish_length);
      app.course.difficulty = app.policy->meta_num("course_difficulty", app.course.difficulty);
    }
    if (!path.empty()) {
      std::fprintf(stderr, "--policy needs a generated course, not a scene file\n");
      return 1;
    }
  }
  if (path.empty()) {
    EnvConfig cfg;
    cfg.course = app.course;
    cfg.episodes_per_course = 1 << 30;  // the viewer changes courses only on request
    cfg.stall_time = cfg.time_limit = 1e9;
    // the policy sees the time left, so it needs the time limit it was trained with
    if (app.policy) cfg.time_limit = app.policy->meta_num("env_time_limit", 0);
    app.env = std::make_unique<ParkourEnv>(cfg);
    app.sim = &app.env->sim();
    for (int k = 0; k < start_section; ++k) jump_section(+1);
    if (app.policy) {
      if (app.policy->obs_size() != app.env->obs_size() || app.policy->act_size() != app.sim->model()->nu) {
        std::fprintf(stderr, "policy does not fit this build (obs %d vs %d, actions %d vs %d)\n",
                     app.policy->obs_size(), app.env->obs_size(), app.policy->act_size(), int(app.sim->model()->nu));
        return 1;
      }
      app.action.assign(app.sim->model()->nu, 0.0);
      app.mode = Mode::kPolicy;
    }
  } else {
    app.fixed_scene = std::make_unique<Simulation>(path);
    app.sim = app.fixed_scene.get();
  }

  if (!glfwInit()) {
    std::fprintf(stderr, "could not initialise GLFW (is a display available?)\n");
    return 1;
  }
  glfwWindowHint(GLFW_SAMPLES, 4);
  GLFWwindow* window = glfwCreateWindow(1400, 900, "Parcour bot", nullptr, nullptr);
  if (!window) {
    std::fprintf(stderr, "could not create window\n");
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjv_defaultCamera(&app.cam);
  mjv_defaultOption(&app.opt);
  mjv_defaultScene(&app.scn);
  mjr_defaultContext(&app.con);
  mjv_defaultPerturb(&app.pert);
  mjv_makeScene(app.sim->model(), &app.scn, 4000);
  mjr_makeContext(app.sim->model(), &app.con, mjFONTSCALE_150);

  app.cam.type = mjCAMERA_TRACKING;
  app.cam.trackbodyid = app.sim->layout().root_body;
  app.cam.distance = 4.0;
  app.cam.azimuth = 120;
  app.cam.elevation = -15;

  glfwSetKeyCallback(window, on_key);
  glfwSetMouseButtonCallback(window, on_button);
  glfwSetCursorPosCallback(window, on_move);
  glfwSetScrollCallback(window, on_scroll);

  reset(0.0);
  if (overview) overview_camera();
  while (!glfwWindowShouldClose(window)) {
    physics();
    mjrRect vp = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &vp.width, &vp.height);
    mjv_updateScene(app.sim->model(), app.sim->data(), &app.opt, &app.pert, &app.cam, mjCAT_ALL,
                    &app.scn);
    decorate();
    mjr_render(vp, &app.scn, &app.con);
    overlay(vp);
    if (screenshot && app.sim->data()->time >= shot_time) {
      save_ppm(screenshot, vp);
      break;
    }
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  mjv_freeScene(&app.scn);
  mjr_freeContext(&app.con);
  glfwTerminate();
  return 0;
}
