"""Training stages: what the bot practises, how it is rewarded and when it moves on.

Each stage sets the course and the reward / termination settings of parcour.EnvConfig.
Training starts at the first stage and advances automatically once the goal holds over the
last WINDOW episodes (after at least MIN_STEPS steps in the stage). The obstacle stages of the
curriculum follow in step 5 of the roadmap.
"""

import parcour

WINDOW = 300         # episodes the goal is measured over
MIN_STEPS = 2_000_000

TRACK = {"num_obstacles": 0, "finish_length": 30.0, "difficulty": 0.0}  # 30 m flat, may turn

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
]


def index(name):
    for i, s in enumerate(STAGES):
        if s["name"] == name:
            return i
    raise ValueError(f"unknown stage {name!r}; stages: {[s['name'] for s in STAGES]}")


def make_config(stage, seed):
    cfg = parcour.EnvConfig()
    cfg.course.seed = seed
    for k, v in stage["course"].items():
        setattr(cfg.course, k, v)
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
    return ", ".join(f"{names[k]} >= {v:g}" for k, v in stage["goal"].items())
