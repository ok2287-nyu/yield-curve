"""Closed-loop evaluation in log replay (CPU). Step 1: the environment's acceptance test and reference policies.

    python scripts/replay_eval.py --cache ../round_cache --data_dir ../round_data --n_windows 300

Policies (controllers will be added here):
    human      the human's exact logged poses: the acceptance test. Expect no front / side collisions (rear and other
               overlaps would be labelling noise in rounD's boxes; their rate is the floor for every other policy)
    human_acc  the human's logged accelerations integrated along the path: the drift of the kinematic integration
    brake      -3 m/s^2: stops, so the logged followers drive into it (rear: the replay's known artefact)
    coast      0 m/s^2
    go         +1.5 m/s^2: drives into whatever is ahead (front)
Episodes start at the end of each validation window's warm-up and run --horizon steps (default 30 = 6 s), stopping
at the first collision. "traffic" = windows where the human ever had a road user less than --near m ahead in its lane.
Writes replay_eval.csv (one row per window and policy). Rendered rounD data: HPC / your own machine only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from roundabout_wm.config import Config
from roundabout_wm.controller.replay import Constant, HumanAccel, HumanExact, ReplayEnv, run_episode

POLICIES = {"human": HumanExact, "human_acc": HumanAccel, "brake": lambda: Constant(-3.0, "brake"),
            "coast": lambda: Constant(0.0, "coast"), "go": lambda: Constant(1.5, "go")}


def summary(df, label):
    print(f"\n[{label}] {df.window.nunique()} windows")
    print(f"  {'policy':10s} {'front':>6s} {'side':>6s} {'rear':>6s} {'gap<2m':>7s} {'minGap':>7s} {'hardBr':>7s} "
          f"{'progr':>6s} {'lag s':>6s} {'freeze':>7s}")
    for name, g in df.groupby("policy", sort=False):
        print(f"  {name:10s} {g.col_front.mean():6.3f} {g.col_side.mean():6.3f} {g.col_rear.mean():6.3f} "
              f"{(g.min_gap < 2).mean():7.3f} {g.min_gap.replace(np.inf, np.nan).median():7.1f} "
              f"{(g.hard_brakes > 0).mean():7.3f} {g.progress.median():6.2f} {g.lag_s.median():6.2f} "
              f"{(g.freeze_s >= 1).mean():7.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=300)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--near", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="replay_eval.csv")
    a = p.parse_args()
    cfg = Config()
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, horizon=a.horizon)
    idx = np.sort(np.random.default_rng(a.seed).choice(len(env.ds), min(a.n_windows, len(env.ds)), replace=False))
    rows = []
    for j, i in enumerate(idx):
        for name in a.policies.split(","):
            pol = POLICIES[name]()
            rows.append(dict(policy=name, **run_episode(env, pol, int(i))))
        if (j + 1) % 25 == 0:
            print(f"  {j + 1}/{len(idx)} windows", end="\r")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    hum = df[df.policy == "human"].set_index("window")
    traffic = hum.index[hum.min_gap < a.near]
    summary(df, "all")
    summary(df[df.window.isin(traffic)], f"traffic: human had a road user < {a.near:g} m ahead")
    bad = hum[hum.col_front | hum.col_side]
    if len(bad):
        print(f"\nacceptance: the human collides (front/side) in {len(bad)} windows: {bad.index.tolist()[:20]}")
    else:
        print("\nacceptance: the human never collides front/side")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
