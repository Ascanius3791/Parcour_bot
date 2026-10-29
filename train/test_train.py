"""Quick end-to-end check of the trainer (about a minute): a few PPO iterations, export,
C++ policy == PyTorch actor, resume keeps the state. Used by CI; not a training run."""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(f"  [{' ok ' if ok else 'FAIL'}] {what}", flush=True)
    failures += not ok


def run(out, extra):
    sys.argv = ["train.py", "--minutes", "0.05", "--envs", "8", "--horizon", "16", "--threads", "2",
                "--eval-envs", "2", "--out", out] + extra
    t = train.Trainer(train.parse_args())
    t.train()
    return t


for st in train.stages.STAGES:
    names = st["course"].get("allowed")
    if names:
        got = train.stages.make_config(st, 1).course.allowed
        check(sorted(got) == sorted(names), f"stage {st['name']}: obstacle names are known")

with tempfile.TemporaryDirectory() as tmp:
    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    t = run(a, [])
    for f in ("ckpt.pt", "policy.bin", "metrics.csv", "eval.json", "summary.md"):
        check(os.path.exists(os.path.join(a, f)), f"writes {f}")

    pol = train.parcour.Policy(os.path.join(a, "policy.bin"))
    obs = t.env.reset()
    with torch.no_grad():
        ref = t.model.actor(t.observe(obs, update=False)).clamp(-1, 1).numpy()
    got = np.stack([pol.act(o) for o in obs])
    err = np.abs(got - ref).max()
    check(err < 1e-4, f"exported C++ policy matches the PyTorch actor (max diff {err:.1e})")
    check(pol.meta["stage"] == "stand" and pol.obs_size == t.env.obs_size, "policy meta data")

    t2 = run(b, ["--resume", a])
    check(t2.state["rounds"] == 2 and t2.state["steps"] > t.state["steps"], "resume continues the counters")
    with open(os.path.join(b, "metrics.csv")) as f:
        rows = f.read().strip().splitlines()
    check(len(rows) >= 2, "metrics continue across resumes")
    t3 = run(os.path.join(tmp, "c"), ["--resume", b, "--stage", "walk"])
    check(t3.stage["name"] == "walk" and t3.state["stage_steps"] < t3.state["steps"], "a stage can be forced")

print("ALL PASSED" if not failures else f"FAILED ({failures})")
sys.exit(1 if failures else 0)
