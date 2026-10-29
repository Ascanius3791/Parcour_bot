#!/usr/bin/env python3
"""Generate the MJCF model of the parkour humanoid.

Conventions
-----------
* Z is up, the body faces +X, its left side is +Y.
* In the default pose (all joint angles 0) every body frame is aligned with the
  world frame; the humanoid stands upright with arms hanging, palms facing the
  thighs.
* Every joint is a hinge. Joint names end in the motion that a POSITIVE angle
  produces (e.g. ``knee_l_flex``: positive = knee bends).
* Segment masses follow de Leva (1996) for a male adult; segment lengths follow
  standard anthropometric ratios of body height.
* Each hand has five fingers (15 passive hinges) driven by a single "grip"
  tendon actuator per hand.

Actuator ``ctrlrange`` is the torque limit in Nm (asymmetric where human
strength is asymmetric). For the grip tendon it is the torque applied at each
finger joint. The environment maps policy actions in [-1, 1] onto these ranges.

Usage: python3 tools/gen_humanoid.py [--height 1.75] [--mass 70] [-o assets/humanoid.xml]
"""
import argparse
import math
import numpy as np

REF_HEIGHT = 1.75

# de Leva (1996), male, fraction of total body mass
MASS_FRACTION = {
    "pelvis": 0.1117, "abdomen": 0.1633, "chest": 0.1596, "head": 0.0694,
    "upper_arm": 0.0271, "forearm": 0.0162, "hand": 0.0061,
    "thigh": 0.1416, "shank": 0.0433, "foot": 0.0110, "toes": 0.0027,
}

# collision filtering: world = 1, body = 2, fingers = 4
BODY_COL = 'contype="2" conaffinity="3"'
FINGER_COL = 'contype="4" conaffinity="1"'

ACTUATORS = []   # (name, joint/tendon, kind, lo, hi)
GRIP_TENDONS = []


def fmt(v):
    return " ".join(f"{x:.4g}" for x in v)


def capsule_volume(r, length):
    return math.pi * r * r * length + 4.0 / 3.0 * math.pi * r ** 3


def box_volume(h):
    return 8.0 * h[0] * h[1] * h[2]


def sphere_volume(r):
    return 4.0 / 3.0 * math.pi * r ** 3


class Body:
    """Tiny builder for an MJCF body with geoms that share the segment mass."""

    def __init__(self, name, pos, mass):
        self.name, self.pos, self.mass = name, pos, mass
        self.joints, self.geoms, self.children, self.extra = [], [], [], []

    def joint(self, name, axis, lo, hi, kind="big", act=None):
        self.joints.append((name, axis, lo, hi, kind))
        if act is not None:
            ACTUATORS.append((name, name, "joint", act[0], act[1]))
        return self

    def capsule(self, a, b, r, col=BODY_COL, rgba=None):
        length = float(np.linalg.norm(np.subtract(b, a)))
        self.geoms.append(("capsule", dict(fromto=fmt(list(a) + list(b)), size=f"{r:.4g}"),
                           capsule_volume(r, length), col, rgba))
        return self

    def box(self, center, half, col=BODY_COL, rgba=None):
        self.geoms.append(("box", dict(pos=fmt(center), size=fmt(half)), box_volume(half), col, rgba))
        return self

    def sphere(self, center, r, col=BODY_COL, rgba=None):
        self.geoms.append(("sphere", dict(pos=fmt(center), size=f"{r:.4g}"), sphere_volume(r), col, rgba))
        return self

    def add(self, child):
        self.children.append(child)
        return child

    def xml(self, indent):
        pad = " " * indent
        cls = ' childclass="humanoid"' if self.name == "pelvis" else ""
        out = [f'{pad}<body name="{self.name}" pos="{fmt(self.pos)}"{cls}>']
        if self.name == "pelvis":
            out.append(f'{pad}  <freejoint name="root"/>')
            out.append(f'{pad}  <camera name="track" mode="trackcom" pos="0 -4 1.2" xyaxes="1 0 0 0 0.29 1"/>')
        for name, axis, lo, hi, kind in self.joints:
            out.append(f'{pad}  <joint name="{name}" class="{kind}" axis="{fmt(axis)}" range="{lo:.4g} {hi:.4g}"/>')
        total_vol = sum(g[2] for g in self.geoms)
        for i, (gtype, attrs, vol, col, rgba) in enumerate(self.geoms):
            mass = self.mass * vol / total_vol
            a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            name = self.name if i == 0 else f"{self.name}_{i}"
            color = f' rgba="{rgba}"' if rgba else ""
            out.append(f'{pad}  <geom name="{name}" type="{gtype}" {a} mass="{mass:.5g}" {col}{color}/>')
        out.extend(f"{pad}  {e}" for e in self.extra)
        for c in self.children:
            out.extend(c.xml(indent + 2))
        out.append(f"{pad}</body>")
        return out


def build(height, mass):
    s = height / REF_HEIGHT
    M = {k: v * mass for k, v in MASS_FRACTION.items()}
    S = lambda *v: [x * s for x in v]  # scale lengths

    # landmark heights (standing, shoes on)
    z_ankle, z_knee, z_hip, z_pelvis = 0.09 * s, 0.52 * s, 0.94 * s, 0.97 * s
    z_lumbar, z_thorax, z_shoulder, z_neck = 1.06 * s, 1.22 * s, 1.43 * s, 1.50 * s

    pelvis = Body("pelvis", [0, 0, z_pelvis], M["pelvis"])
    pelvis.capsule(S(0, -0.06, 0), S(0, 0.06, 0), 0.095 * s)
    # vestibular system / IMU lives in the head, but a pelvis site is handy for sensors
    pelvis.extra.append('<site name="pelvis_imu" pos="0 0 0" size="0.01"/>')

    abdomen = pelvis.add(Body("abdomen", [0, 0, z_lumbar - z_pelvis], M["abdomen"]))
    abdomen.joint("lumbar_flex", [0, 1, 0], -25, 60, act=(-250, 200))
    abdomen.joint("lumbar_bend", [1, 0, 0], -25, 25, act=(-150, 150))
    abdomen.joint("lumbar_twist", [0, 0, 1], -25, 25, act=(-80, 80))
    abdomen.capsule(S(0, -0.05, 0.08), S(0, 0.05, 0.08), 0.09 * s)

    chest = abdomen.add(Body("chest", [0, 0, z_thorax - z_lumbar], M["chest"]))
    chest.joint("thorax_flex", [0, 1, 0], -20, 40, act=(-150, 120))
    chest.joint("thorax_bend", [1, 0, 0], -20, 20, act=(-100, 100))
    chest.joint("thorax_twist", [0, 0, 1], -35, 35, act=(-60, 60))
    chest.capsule(S(0, -0.08, 0.10), S(0, 0.08, 0.10), 0.11 * s)
    chest.capsule(S(0, -0.09, 0.20), S(0, 0.09, 0.20), 0.075 * s)

    head = chest.add(Body("head", [0, 0, z_neck - z_thorax], M["head"]))
    head.joint("neck_flex", [0, 1, 0], -50, 60, kind="mid", act=(-40, 40))
    head.joint("neck_bend", [1, 0, 0], -40, 40, kind="mid", act=(-30, 30))
    head.joint("neck_twist", [0, 0, 1], -70, 70, kind="mid", act=(-20, 20))
    head.capsule(S(0, 0, 0.0), S(0, 0, 0.05), 0.05 * s)
    head.sphere(S(0.01, 0, 0.15), 0.10 * s)
    head.extra.append('<site name="head_imu" pos="0 0 0.15" size="0.01"/>')

    for side, sgn in (("l", 1), ("r", -1)):
        # ------------------------------------------------------------- arm
        upper = chest.add(Body(f"upper_arm_{side}", [0, 0.20 * sgn * s, z_shoulder - z_thorax],
                               M["upper_arm"]))
        upper.joint(f"shoulder_{side}_flex", [0, -1, 0], -60, 180, act=(-120, 100))
        upper.joint(f"shoulder_{side}_abd", [sgn, 0, 0], -30, 180, act=(-130, 100))
        upper.joint(f"shoulder_{side}_rot", [0, 0, sgn], -90, 90, act=(-60, 60))
        upper.capsule(S(0, 0, -0.03), S(0, 0, -0.28), 0.045 * s)

        fore = upper.add(Body(f"forearm_{side}", S(0, 0, -0.30), M["forearm"]))
        fore.joint(f"elbow_{side}_flex", [0, -1, 0], -5, 150, act=(-70, 90))
        fore.joint(f"elbow_{side}_pron", [0, 0, sgn], -85, 80, kind="mid", act=(-15, 15))
        fore.capsule(S(0, 0, -0.02), S(0, 0, -0.24), 0.038 * s)

        hand = fore.add(Body(f"hand_{side}", S(0, 0, -0.26), M["hand"] * 0.7))
        hand.joint(f"wrist_{side}_flex", [-sgn, 0, 0], -70, 80, kind="mid", act=(-20, 20))
        hand.joint(f"wrist_{side}_dev", [0, -1, 0], -35, 20, kind="mid", act=(-15, 15))
        hand.box(S(0, 0, -0.05), S(0.042, 0.015, 0.05))

        finger_mass = M["hand"] * 0.3 / 15.0
        flex_axis = [-sgn, 0, 0]  # rotates fingertip towards the palm side
        grip_joints = []
        # x offset, phalanx lengths (proximal, middle, distal), radius
        fingers = [("index", 0.032, (0.045, 0.027, 0.022), 0.0085),
                   ("middle", 0.011, (0.050, 0.030, 0.023), 0.0090),
                   ("ring", -0.011, (0.047, 0.028, 0.022), 0.0085),
                   ("little", -0.031, (0.037, 0.022, 0.020), 0.0075)]
        for fname, x, lens, r in fingers:
            parent, pos = hand, S(x, 0, -0.10)
            for k, (L, lo, hi) in enumerate(zip(lens, (-10, 0, 0), (90, 100, 80))):
                ph = parent.add(Body(f"{fname}{k + 1}_{side}", pos, finger_mass))
                jname = f"{fname}{k + 1}_{side}_flex"
                ph.joint(jname, flex_axis, lo, hi, kind="finger")
                ph.capsule([0, 0, 0], S(0, 0, -(L - r)), r * s, col=FINGER_COL)
                grip_joints.append(jname)
                parent, pos = ph, S(0, 0, -L)

        # thumb: sits at the front edge of the palm, points down/forward/across the palm
        d = np.array([0.45, -0.45 * sgn, -0.75])
        d /= np.linalg.norm(d)
        target = np.array([-1.0, -0.4 * sgn, 0.0])
        axis = np.cross(d, target)
        axis /= np.linalg.norm(axis)
        parent, pos = hand, S(0.04, -0.012 * sgn, -0.025)
        for k, (L, r) in enumerate(((0.040, 0.011), (0.032, 0.0095), (0.026, 0.0085))):
            ph = parent.add(Body(f"thumb{k + 1}_{side}", pos, finger_mass))
            jname = f"thumb{k + 1}_{side}_flex"
            ph.joint(jname, list(axis), 0, 60 if k == 0 else 80, kind="finger")
            ph.capsule([0, 0, 0], list(d * (L - r) * s), r * s, col=FINGER_COL)
            grip_joints.append(jname)
            parent, pos = ph, list(d * L * s)

        ACTUATORS.append((f"grip_{side}", f"grip_{side}", "tendon", -1.0, 4.0))
        GRIP_TENDONS.append((f"grip_{side}", grip_joints))

        # ------------------------------------------------------------- leg
        thigh = pelvis.add(Body(f"thigh_{side}", [0, 0.09 * sgn * s, z_hip - z_pelvis], M["thigh"]))
        thigh.joint(f"hip_{side}_flex", [0, -1, 0], -30, 125, act=(-280, 180))
        thigh.joint(f"hip_{side}_abd", [sgn, 0, 0], -30, 50, act=(-150, 150))
        thigh.joint(f"hip_{side}_rot", [0, 0, sgn], -35, 45, act=(-80, 80))
        thigh.capsule(S(0, 0, -0.02), S(0, 0, -0.38), 0.068 * s)

        shank = thigh.add(Body(f"shank_{side}", [0, 0, z_knee - z_hip], M["shank"]))
        shank.joint(f"knee_{side}_flex", [0, 1, 0], -5, 150, act=(-280, 150))
        shank.capsule(S(0, 0, -0.02), S(0, 0, -0.38), 0.05 * s)

        foot = shank.add(Body(f"foot_{side}", [0, 0, z_ankle - z_knee], M["foot"]))
        foot.joint(f"ankle_{side}_dorsi", [0, -1, 0], -50, 25, act=(-250, 60))
        foot.joint(f"ankle_{side}_inv", [-sgn, 0, 0], -20, 30, act=(-60, 60))
        foot.box(S(0.035, 0, -0.055), S(0.10, 0.05, 0.035), rgba="0.15 0.15 0.18 1")
        foot.extra.append(f'<site name="sole_{side}" pos="{fmt(S(0.035, 0, -0.09))}" '
                          f'size="{fmt(S(0.10, 0.05, 0.005))}" type="box" rgba="0 0 0 0"/>')

        toes = foot.add(Body(f"toes_{side}", S(0.135, 0, -0.065), M["toes"]))
        toes.joint(f"toe_{side}_dorsi", [0, -1, 0], -30, 60, kind="mid", act=(-25, 10))
        toes.box(S(0.03, 0, 0), S(0.03, 0.05, 0.025), rgba="0.15 0.15 0.18 1")
        toes.extra.append(f'<site name="toesole_{side}" pos="{fmt(S(0.03, 0, -0.025))}" '
                          f'size="{fmt(S(0.03, 0.05, 0.005))}" type="box" rgba="0 0 0 0"/>')

    return pelvis


def emit(height, mass):
    ACTUATORS.clear()
    GRIP_TENDONS.clear()
    root = build(height, mass)
    L = []
    L.append(f"<!-- Generated by tools/gen_humanoid.py (height={height} m, mass={mass} kg). Do not edit. -->")
    L.append('<mujoco model="parcour_humanoid">')
    L.append("  <default>")
    L.append('    <default class="humanoid">')
    L.append('      <geom friction="1.0 0.02 0.002" solref="0.005 1" condim="3" rgba="0.85 0.68 0.55 1"/>')
    L.append('      <joint type="hinge" limited="true" solreflimit="0.004 1" solimplimit="0.95 0.99 0.001"/>')
    L.append('      <motor ctrllimited="true"/>')
    L.append('      <default class="big"><joint damping="2" armature="0.02"/></default>')
    L.append('      <default class="mid"><joint damping="0.5" armature="0.005"/></default>')
    L.append('      <default class="finger"><joint damping="0.02" armature="0.0005" stiffness="0.05"/></default>')
    L.append("    </default>")
    L.append("  </default>")
    L.append("")
    L.append('  <worldbody>')
    L.extend(root.xml(4))
    L.append("  </worldbody>")
    L.append("")
    L.append("  <tendon>")
    for name, joints in GRIP_TENDONS:
        L.append(f'    <fixed name="{name}">')
        L.extend(f'      <joint joint="{j}" coef="1"/>' for j in joints)
        L.append("    </fixed>")
    L.append("  </tendon>")
    L.append("")
    L.append("  <!-- ctrlrange = torque limit in Nm -->")
    L.append('  <actuator>')
    for name, target, kind, lo, hi in ACTUATORS:
        L.append(f'    <motor name="{name}" {kind}="{target}" ctrlrange="{lo:.4g} {hi:.4g}" class="humanoid"/>')
    L.append("  </actuator>")
    L.append("")
    L.append("  <sensor>")
    L.append('    <accelerometer name="head_acc" site="head_imu"/>')
    L.append('    <gyro name="head_gyro" site="head_imu"/>')
    for side in ("l", "r"):
        L.append(f'    <touch name="sole_{side}" site="sole_{side}"/>')
        L.append(f'    <touch name="toesole_{side}" site="toesole_{side}"/>')
    L.append("  </sensor>")
    L.append("</mujoco>")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--mass", type=float, default=70.0)
    ap.add_argument("-o", "--out", default="assets/humanoid.xml")
    args = ap.parse_args()
    with open(args.out, "w") as f:
        f.write(emit(args.height, args.mass))
    print(f"wrote {args.out}: {len(ACTUATORS)} actuators")


if __name__ == "__main__":
    main()
