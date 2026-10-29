"""Picks the best of the parallel training runs of one round and copies it to OUT.

  python train/select_best.py MEMBERS_DIR OUT_DIR

Every subdirectory of MEMBERS_DIR is one run (eval.json, ckpt.pt, policy.bin, ...). The best
is the one at the latest stage and difficulty, then with the highest evaluation return there.
"""

import json
import os
import shutil
import sys


def main(members_dir, out):
    runs = []
    for name in sorted(os.listdir(members_dir)):
        path = os.path.join(members_dir, name)
        try:
            with open(os.path.join(path, "eval.json")) as f:
                ev = json.load(f)
        except (OSError, ValueError):
            print(f"{name}: no evaluation, skipped")
            continue
        if os.path.exists(os.path.join(path, "ckpt.pt")):
            runs.append((name, path, ev))
    if not runs:
        sys.exit("no usable training run")
    best = max(runs, key=lambda r: (r[2]["stage_index"], r[2].get("difficulty", 0), r[2]["goal_reached"],
                                    r[2]["metrics"].get("return", -1e9)))
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(best[1], out)
    lines = ["", f"Parallel runs this round ({len(runs)}), evaluated on the same courses; kept **{best[0]}**:", "",
             "| run | stage | difficulty | return | progress (m) | finished | speed (m/s) |",
             "|---|---|---|---|---|---|---|"]
    for name, _, ev in runs:
        m = ev["metrics"]
        mark = " (kept)" if name == best[0] else ""
        lines.append(f"| {name}{mark} | {ev['stage']} | {ev.get('difficulty', 0):g} | {m.get('return', 0):.1f} | "
                     f"{m.get('progress', 0):.1f} | {100 * m.get('finish_rate', 0):.0f}% | {m.get('speed', 0):.2f} |")
    with open(os.path.join(out, "summary.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"kept {best[0]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
