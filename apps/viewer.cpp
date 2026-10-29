// Interactive 3D viewer for the parkour humanoid.
// Usage: parcour_viewer [scene.xml] [--screenshot out.ppm --at SECONDS]
//
// Shows the body, its centre of mass, momentum and angular momentum, and colours every
// segment by the stress on the joint above it (skin -> red; magenta = broken).
// Press H for the key list.

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/controllers.h"
#include "core/simulation.h"

using namespace parcour;

namespace {

enum class Mode { kLimp, kHoldPose };

struct App {
  Simulation* sim = nullptr;
  mjvCamera cam;
  mjvOption opt;
  mjvScene scn;
  mjrContext con;
  mjvPerturb pert;

  Mode mode = Mode::kHoldPose;
  bool paused = false, step_once = false;
  bool grip = false, invincible = false;
  bool show_com = true, show_limbs = false, show_stress = true, show_help = false;
  bool tracking = true;
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
    "Dbl-click\nCtrl+L drag\nCtrl+R drag\nH";
const char* kHelpText =
    "pause / run\nsingle step (paused)\nreset standing\ndrop from height\n"
    "mode: limp / hold pose\ngrip on / off\ninvincible (no breaking)\ncom + momentum\n"
    "per-limb momentum\nstress colours\ntrack camera\ndrop height\n"
    "select body\nrotate body\npull body\nhelp";

void reset(double lift) {
  app.sim->reset(lift);
  app.sim_wall_offset = glfwGetTime() - app.sim->data()->time;
  mjv_defaultPerturb(&app.pert);
  app.pert.select = 0;
}

void apply_controller() {
  const mjModel* m = app.sim->model();
  mjData* d = app.sim->data();
  if (app.mode == Mode::kHoldPose) {
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
    if (!ok && !app.invincible) {
      app.paused = true;
      break;
    }
  }
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

  if (app.show_com) {
    const float red[4] = {1, 0.2, 0.2, 1}, yellow[4] = {1, 0.85, 0.1, 1},
                blue[4] = {0.2, 0.5, 1, 1}, grey[4] = {1, 0.2, 0.2, 0.6};
    add_sphere(s.com, 0.035, red);
    add_line(s.com, {s.com.x, s.com.y, 0}, grey);
    add_sphere({s.com.x, s.com.y, 0.003}, 0.02, grey);
    add_arrow(s.com, s.momentum * 0.005, 0.02, yellow);    // 1 m per 200 kg m/s
    add_arrow(s.com, s.ang_momentum * 0.05, 0.02, blue);   // 1 m per 20 kg m^2/s
  }
  if (app.show_limbs) {
    const float green[4] = {0.2, 0.9, 0.3, 1};
    for (const LimbState& L : s.limbs) add_arrow(s.com + L.pos, L.momentum * 0.02, 0.008, green);
  }
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
                app.mode == Mode::kLimp ? "limp" : "hold pose", app.invincible ? " (invincible)" : "",
                app.grip ? "closed" : "open", s.com.z, s.com_vel.norm(), s.momentum.norm(),
                s.ang_momentum.norm(), 100 * inj.loads()[worst].stress,
                InjuryMonitor::joint_name(worst).c_str());
  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, left, right, &app.con);

  if (inj.broken_segment() >= 0) {
    std::string msg = "JOINT BROKEN: " + InjuryMonitor::joint_name(inj.broken_segment());
    mjr_overlay(mjFONT_BIG, mjGRID_TOP, vp, msg.c_str(),
                app.invincible ? nullptr : "Backspace = reset, I = continue invincible", &app.con);
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
    case GLFW_KEY_M: app.mode = app.mode == Mode::kLimp ? Mode::kHoldPose : Mode::kLimp; break;
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
  std::string path = std::string(PARCOUR_ASSET_DIR) + "/scene_flat.xml";
  const char* screenshot = nullptr;
  double shot_time = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--screenshot") && i + 1 < argc) screenshot = argv[++i];
    else if (!std::strcmp(argv[i], "--at") && i + 1 < argc) shot_time = std::atof(argv[++i]);
    else path = argv[i];
  }
  Simulation sim(path);
  app.sim = &sim;
  const mjModel* m = sim.model();

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
  mjv_makeScene(m, &app.scn, 4000);
  mjr_makeContext(m, &app.con, mjFONTSCALE_150);

  app.cam.type = mjCAMERA_TRACKING;
  app.cam.trackbodyid = sim.layout().root_body;
  app.cam.distance = 4.0;
  app.cam.azimuth = 120;
  app.cam.elevation = -15;

  glfwSetKeyCallback(window, on_key);
  glfwSetMouseButtonCallback(window, on_button);
  glfwSetCursorPosCallback(window, on_move);
  glfwSetScrollCallback(window, on_scroll);

  reset(0.0);
  while (!glfwWindowShouldClose(window)) {
    physics();
    mjrRect vp = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &vp.width, &vp.height);
    mjv_updateScene(m, sim.data(), &app.opt, &app.pert, &app.cam, mjCAT_ALL, &app.scn);
    decorate();
    mjr_render(vp, &app.scn, &app.con);
    overlay(vp);
    if (screenshot && sim.data()->time >= shot_time) {
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
