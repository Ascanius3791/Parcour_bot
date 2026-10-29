"""Trains the parkour policy with PPO.

  python train/train.py --minutes 60 --out runs/a              # fresh start
  python train/train.py --minutes 60 --out runs/b --resume runs/a

The run stops after --minutes, then evaluates the policy on fixed courses and writes into
--out: ckpt.pt (everything needed to continue), policy.bin (for the viewer: parcour_viewer
--policy policy.bin), metrics.csv (the learning curve, continued across resumes), eval.json
and summary.md.
"""

import argparse
import collections
import csv
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parcour  # noqa: E402  (built by CMake into build/)
import stages  # noqa: E402
from ppo import PPO, ActorCritic, RewardScaler, RunningNorm, export_policy  # noqa: E402

EVAL_SEED = 900_000_000
LOG_EVERY = 10  # iterations per metrics row
SAVE_EVERY = 15 * 60  # seconds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=60, help="training time before evaluating and saving")
    p.add_argument("--out", default="runs/local")
    p.add_argument("--resume", help="directory (or ckpt.pt) of a previous run")
    p.add_argument("--stage", help="force this stage (default: stand, or where the resumed run was)")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=64, help="steps per env per iteration")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-envs", type=int, default=32)
    return p.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.state = {"stage": 0, "steps": 0, "stage_steps": 0, "iteration": 0, "hours": 0.0,
                      "rounds": 0, "history": []}
        ckpt = None
        if args.resume:
            path = args.resume if args.resume.endswith(".pt") else os.path.join(args.resume, "ckpt.pt")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.state.update(ckpt["state"])
            src = os.path.join(os.path.dirname(path), "metrics.csv")
            if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(os.path.join(args.out, "metrics.csv")):
                os.makedirs(args.out, exist_ok=True)
                shutil.copy(src, os.path.join(args.out, "metrics.csv"))
        if args.stage:
            new = stages.index(args.stage)
            if new != self.state["stage"]:
                self.state["stage"], self.state["stage_steps"] = new, 0
        self.state["rounds"] += 1

        self.env = self.make_env()
        self.segments = self.env.obs_segments
        self.model = ActorCritic(self.segments, self.env.act_size, init_std=0.2)
        self.ppo = PPO(self.model)
        self.obs_norm = RunningNorm(self.env.obs_size)
        self.rew_scale = RewardScaler(args.envs, self.ppo.gamma)
        if ckpt:
            self.model.load_state_dict(ckpt["model"])
            self.ppo.opt.load_state_dict(ckpt["optimizer"])
            self.ppo.lr = ckpt["lr"]
            self.obs_norm.load(ckpt["obs_norm"])
            self.rew_scale.norm.load(ckpt["ret_norm"])
        self.window = collections.deque(maxlen=stages.WINDOW)

    @property
    def stage(self):
        return stages.STAGES[self.state["stage"]]

    def make_env(self):
        seed = (self.args.seed * 1_000_003 + self.state["iteration"] * 7919 + 1) % (1 << 62)
        cfg = stages.make_config(self.stage, seed)
        return parcour.VecEnv(cfg, self.args.envs, self.args.threads)

    def observe(self, raw, update=True):
        if update:
            self.obs_norm.update(raw)
        return torch.from_numpy(self.obs_norm.normalize(raw))

    def rollout(self, obs):
        T, N = self.args.horizon, self.args.envs
        buf = {k: [] for k in ("obs", "act", "logp", "value", "mu", "rew", "done")}
        episodes = []
        for _ in range(T):
            with torch.no_grad():
                dist = self.model.dist(obs)
                act = dist.sample()
                value = self.model.value(obs)
            raw, rew, done, trunc, final = self.env.step(act.numpy())
            rew = self.rew_scale(rew, done)
            if trunc.any():  # time limit: the episode could have gone on, bootstrap its value
                with torch.no_grad():
                    rew[trunc] += self.ppo.gamma * self.model.value(self.observe(final[trunc], update=False)).numpy()
            buf["obs"].append(obs)
            buf["act"].append(act)
            buf["logp"].append(dist.log_prob(act).sum(-1))
            buf["value"].append(value)
            buf["mu"].append(dist.mean)
            buf["rew"].append(torch.from_numpy(rew))
            buf["done"].append(torch.from_numpy(done.astype(np.float32)))
            obs = self.observe(raw)
            episodes += self.env.episodes()
        with torch.no_grad():
            last_value = self.model.value(obs)
        b = {k: torch.stack(v) for k, v in buf.items()}
        adv, ret = self.ppo.advantages(b["rew"], b["value"], b["done"], last_value)
        std = self.model.log_std.detach().exp()
        batch = {"obs": b["obs"].reshape(T * N, -1), "act": b["act"].reshape(T * N, -1),
                 "logp": b["logp"].reshape(-1), "value": b["value"].reshape(-1),
                 "mu": b["mu"].reshape(T * N, -1), "std": std.expand(T * N, -1),
                 "adv": adv.reshape(-1), "ret": ret.reshape(-1)}
        return obs, batch, episodes

    def goal_done(self):
        h = self.state["history"]
        return bool(h) and h[-1]["stage"] == self.stage["name"]

    def check_goal(self):
        """Records the stage goal once it holds and moves on to the next stage (returns True then)."""
        if self.state["stage_steps"] < stages.MIN_STEPS or len(self.window) < stages.WINDOW or self.goal_done():
            return False
        m = stages.episode_metrics(list(self.window))
        if not stages.goal_reached(self.stage, m):
            return False
        self.state["history"].append({"stage": self.stage["name"], "steps": self.state["steps"],
                                      "hours": round(self.state["hours"], 2),
                                      "metrics": {k: round(v, 3) for k, v in m.items()}})
        print(f"*** goal of stage '{self.stage['name']}' reached: {stages.describe_goal(self.stage)}", flush=True)
        if self.state["stage"] + 1 >= len(stages.STAGES):
            print("*** that was the last stage; training continues to improve it", flush=True)
            return False
        self.state["stage"] += 1
        self.state["stage_steps"] = 0
        self.window.clear()
        self.env = self.make_env()
        print(f"*** next stage '{self.stage['name']}': {self.stage['about']}", flush=True)
        return True

    def train(self):
        args = self.args
        os.makedirs(args.out, exist_ok=True)
        start = last_save = time.time()
        deadline = start + args.minutes * 60
        obs = self.observe(self.env.reset())
        log_eps, log_stats, log_t, log_steps = [], [], time.time(), 0
        print(f"stage '{self.stage['name']}' ({self.stage['about']}), {self.env.num_envs} envs on "
              f"{self.env.threads} threads, obs {self.env.obs_size}, actions {self.env.act_size}", flush=True)
        while time.time() < deadline:
            obs, batch, episodes = self.rollout(obs)
            stats = self.ppo.update(batch)
            n = args.envs * args.horizon
            self.state["steps"] += n
            self.state["stage_steps"] += n
            self.state["iteration"] += 1
            self.window.extend(episodes)
            log_eps += episodes
            log_stats.append(stats)
            log_steps += n
            now = time.time()
            if self.state["iteration"] % LOG_EVERY == 0:
                self.state["hours"] += (now - log_t) / 3600
                self.log_row(log_eps, log_stats, log_steps / (now - log_t))
                log_eps, log_stats, log_t, log_steps = [], [], now, 0
            if self.check_goal():
                obs = self.observe(self.env.reset())
                self.rew_scale.resize(args.envs)
            if now - last_save > SAVE_EVERY:
                self.save()
                last_save = now
        if log_stats:
            now = time.time()
            self.state["hours"] += (now - log_t) / 3600
            self.log_row(log_eps, log_stats, log_steps / (now - log_t))
        self.save()
        result = self.evaluate()
        with open(os.path.join(args.out, "eval.json"), "w") as f:
            json.dump(result, f, indent=1)
        self.write_summary(result)
        return result

    def log_row(self, episodes, stats, sps):
        m = stages.episode_metrics(episodes)
        row = {"hours": round(self.state["hours"], 4), "steps": self.state["steps"],
               "stage": self.stage["name"], "iteration": self.state["iteration"], "sps": round(sps)}
        keys = ["episodes", "return", "seconds", "survival", "progress", "finish_rate", "speed"]
        keys += ["end:" + r for r in parcour.end_reasons[1:]]
        for k in keys:
            row[k] = round(m.get(k, float("nan")), 4)
        row["std"] = round(self.model.log_std.exp().mean().item(), 4)
        row["lr"] = self.ppo.lr
        for k in ("kl", "value_loss", "entropy"):
            row[k] = round(sum(s[k] for s in stats) / len(stats), 5)
        path = os.path.join(self.args.out, "metrics.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{row['hours']:6.2f} h {row['steps'] / 1e6:8.2f} M steps {row['sps']:5d}/s  {row['stage']:6s} "
              f"return {row['return']:7.2f}  ep {row['seconds']:5.1f} s  progress {row['progress']:5.1f} m  "
              f"finished {row['finish_rate']:.2f}  speed {row['speed']:.2f}  std {row['std']:.3f}  "
              f"lr {row['lr']:.1e}  kl {row['kl']:.4f}", flush=True)

    def meta(self):
        st = self.stage
        meta = {"stage": st["name"], "steps": self.state["steps"], "hours": round(self.state["hours"], 2)}
        for k, v in st["course"].items():
            meta["course_" + k] = v
        meta["env_time_limit"] = st["env"].get("time_limit", 0)
        return meta

    def save(self):
        out = self.args.out
        torch.save({"state": self.state, "model": self.model.state_dict(), "optimizer": self.ppo.opt.state_dict(),
                    "lr": self.ppo.lr, "obs_norm": self.obs_norm.state(), "ret_norm": self.rew_scale.norm.state(),
                    "segments": self.segments}, os.path.join(out, "ckpt.tmp"))
        os.replace(os.path.join(out, "ckpt.tmp"), os.path.join(out, "ckpt.pt"))
        export_policy(os.path.join(out, "policy.bin"), self.model, self.obs_norm, self.meta())

    def evaluate(self):
        """Deterministic policy, one episode on each of eval_envs fixed courses of the current stage."""
        cfg = stages.make_config(self.stage, EVAL_SEED)
        env = parcour.VecEnv(cfg, self.args.eval_envs, self.args.threads)
        obs = env.reset()
        first = {}
        max_steps = int(self.stage["env"].get("time_limit", 120) / 0.02) + 10
        for _ in range(max_steps):
            with torch.no_grad():
                act = self.model.actor(self.observe(obs, update=False))
            obs, _, _, _, _ = env.step(act.clamp(-1, 1).numpy())
            for e in env.episodes():
                first.setdefault(e["env"], e)
            if len(first) == env.num_envs:
                break
        m = stages.episode_metrics(list(first.values()))
        return {"stage": self.stage["name"], "stage_index": self.state["stage"], "steps": self.state["steps"],
                "hours": self.state["hours"], "goal_reached": stages.goal_reached(self.stage, m),
                "metrics": m}

    def write_summary(self, result):
        st, s = self.stage, self.state
        m = result["metrics"]
        lines = [f"## Parcour bot training, round {s['rounds']}", "",
                 f"**Stage {s['stage'] + 1} of {len(stages.STAGES)}: {st['name']}** ({st['about']}). "
                 f"Goal: {stages.describe_goal(st)}.", "",
                 f"Trained so far: {s['steps'] / 1e6:.1f} M steps in {s['hours']:.1f} h "
                 f"({s['steps'] * 0.02 / 3600:.1f} h of simulated time).", ""]
        if s["history"]:
            lines += ["| goal reached | after | hours |", "|---|---|---|"]
            lines += [f"| {h['stage']} | {h['steps'] / 1e6:.1f} M steps | {h['hours']:.1f} |" for h in s["history"]]
            lines.append("")
        lines += [f"Evaluation on {m.get('episodes', 0)} fixed courses (no exploration noise):", "",
                  "| return | seconds survived | progress (m) | finished | speed (m/s) |", "|---|---|---|---|---|",
                  f"| {m.get('return', 0):.1f} | {m.get('seconds', 0):.1f} | {m.get('progress', 0):.1f} | "
                  f"{100 * m.get('finish_rate', 0):.0f}% | {m.get('speed', 0):.2f} |", ""]
        ends = ", ".join(f"{k[4:]} {100 * v:.0f}%" for k, v in m.items() if k.startswith("end:") and v > 0)
        lines += [f"How the evaluation runs ended: {ends or 'n/a'}.", ""]
        path = os.path.join(self.args.out, "metrics.csv")
        if os.path.exists(path):
            with open(path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                lines += ["Learning curve (training episodes, with exploration noise):", "",
                          "| hours | M steps | stage | return | seconds | progress (m) | finished | speed |",
                          "|---|---|---|---|---|---|---|---|"]
                pick, next_h = [], 0.0
                for r in rows:  # about one row per half hour, plus the last one
                    if float(r["hours"]) >= next_h:
                        pick.append(r)
                        next_h = float(r["hours"]) + 0.5
                if pick[-1] is not rows[-1]:
                    pick.append(rows[-1])
                for r in pick:
                    lines.append(f"| {float(r['hours']):.1f} | {int(r['steps']) / 1e6:.1f} | {r['stage']} | "
                                 f"{float(r['return']):.1f} | {float(r['seconds']):.1f} | {float(r['progress']):.1f} | "
                                 f"{100 * float(r['finish_rate']):.0f}% | {float(r['speed']):.2f} |")
        with open(os.path.join(self.args.out, "summary.md"), "w") as f:
            f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    result = Trainer(args).train()
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
