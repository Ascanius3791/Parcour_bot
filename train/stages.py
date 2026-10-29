"""Training stages: what the bot practises, how it is rewarded and when it moves on.

Each stage sets the course and the reward / termination settings of parcour.EnvConfig.
Training starts at the first stage and advances automatically once the goal holds over the
last WINDOW episodes (after at least MIN_STEPS steps in the stage).

The obstacle stages start at difficulty 0 and raise it by RAMP_STEP whenever RAMP_FINISH of
the last WINDOW episodes finished, up to the stage's "ramp" value; their goal counts only
once that difficulty is reached. Half of their episodes start in front of a random obstacle,
so the later obstacles of a course get as much practice as the first.
"""

import parcour

WINDOW = 300         # episodes the goal is measured over
MIN_STEPS = 2_000_000

RAMP_STEP = 0.1
RAMP_FINISH = 0.6

TRACK = {"num_obstacles": 0, "finish_length": 30.0, "difficulty": 0.0}  # 30 m flat, may turn

# rewards as in "run", with a smaller bonus per second left: the time limit is longer here
# (10 s + 1.5 s per metre of course); a run ends when the com gets this low above the surface
OBSTACLE_ENV = {"w_progress": 1.0, "w_alive": 0.1, "w_action": 0.02, "down_height": 0.4,
                "stall_time": 4.0, "time_limit": 0.0, "init_noise": 0.05, "finish_bonus": 5.0,
                "finish_time_bonus": 0.3, "random_start": 0.5}


def obstacle_stage(name, about, allowed, obstacles, ramp, finish_rate=0.6, **env):
    return {"name": name, "about": about,
            "course": {"num_obstacles": obstacles, "finish_length": 4.0, "difficulty": 0.0, "allowed": allowed},
            "env": {**OBSTACLE_ENV, **env}, "ramp": ramp, "goal": {"finish_rate": finish_rate}}


TERRAIN = ["stairs", "hurdle", "platforms"]

STAGES = [
    {
        "name": "stand",
        "about": "stay on your feet for 10 s",
        "course": TRACK,
        "env": {"w_progress": 0.0, "w_alive": 1.0, "w_action": 0.1, "down_height": 0.75,
                "stall_time": 1e9, "time_limit": 10.0, "init_noise": 0.05, "finish_bonus": 0.0},
        "goal": {"survival": 0.9},
    },
    {
        "name": "walk",
        "about": "reach the end of a 30 m track within 45 s",
        "course": TRACK,
        "env": {"w_progress": 1.0, "w_alive": 0.5, "w_action": 0.05, "down_height": 0.65,
                "stall_time": 4.0, "time_limit": 45.0, "init_noise": 0.05, "finish_bonus": 5.0},
        "goal": {"finish_rate": 0.7},
    },
    {
        "name": "run",
        "about": "cover the 30 m track at 3 m/s",
        "course": TRACK,
        "env": {"w_progress": 1.0, "w_alive": 0.1, "w_action": 0.02, "down_height": 0.6,
                "stall_time": 3.0, "time_limit": 25.0, "init_noise": 0.05, "finish_bonus": 5.0,
                "finish_time_bonus": 1.0},
        "goal": {"finish_rate": 0.7, "speed": 3.0},
    },
    obstacle_stage("terrain", "stairs, hurdles and platforms, 3 per course", TERRAIN, 3, ramp=0.5),
    obstacle_stage("beam", "adds balance beams over pits, 4 obstacles", TERRAIN + ["beam"], 4, ramp=0.5),
    # crouching through low holes brings the com down
    obstacle_stage("wall", "adds walls with a hole to get through", TERRAIN + ["beam", "wall"], 4, ramp=0.5,
                   down_height=0.25),
    obstacle_stage("bars", "adds swing bars over pits", TERRAIN + ["beam", "wall", "bars"], 5, ramp=0.5,
                   down_height=0.25),
    obstacle_stage("parkour", "all obstacles, 6 per course, up to the hardest", TERRAIN + ["beam", "wall", "bars"],
                   6, ramp=1.0, finish_rate=0.5, down_height=0.25),
]


def index(name):
    for i, s in enumerate(STAGES):
        if s["name"] == name:
            return i
    raise ValueError(f"unknown stage {name!r}; stages: {[s['name'] for s in STAGES]}")


def make_config(stage, seed, difficulty=None):
    """EnvConfig of a stage; difficulty overrides the stage's starting difficulty."""
    cfg = parcour.EnvConfig()
    cfg.course.seed = seed
    for k, v in stage["course"].items():
        setattr(cfg.course, k, v)
    if difficulty is not None:
        cfg.course.difficulty = difficulty
    for k, v in stage["env"].items():
        setattr(cfg, k, v)
    cfg.action_mode = parcour.ActionMode.target
    return cfg


def episode_metrics(episodes):
    """Summary of a list of episode dicts from VecEnv.episodes()."""
    if not episodes:
        return {}
    n = len(episodes)
    finished = [e for e in episodes if e["finished"]]
    out = {
        "episodes": n,
        "return": sum(e["return"] for e in episodes) / n,
        "seconds": sum(e["time"] for e in episodes) / n,
        "survival": sum(min(1.0, e["time"] / e["time_limit"]) for e in episodes) / n,
        "progress": sum(e["progress"] for e in episodes) / n,
        "finish_rate": len(finished) / n,
        "speed": (sum(e["progress"] / max(e["time"], 1e-6) for e in finished) / len(finished)) if finished else 0.0,
    }
    for r in parcour.end_reasons[1:]:
        out["end:" + r] = sum(e["reason"] == r for e in episodes) / n
    return out


def goal_reached(stage, metrics):
    return all(metrics.get(k, 0.0) >= v for k, v in stage["goal"].items())


def describe_goal(stage):
    names = {"survival": "time survived", "finish_rate": "finish rate", "speed": "speed (m/s)"}
    goal = ", ".join(f"{names[k]} >= {v:g}" for k, v in stage["goal"].items())
    return f"difficulty {stage['ramp']:g} reached, {goal}" if "ramp" in stage else goal
