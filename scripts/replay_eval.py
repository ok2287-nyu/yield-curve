"""Closed-loop evaluation in log replay (CPU). Step 1: the environment's acceptance test and reference policies.

    python scripts/replay_eval.py --cache ../round_cache --data_dir ../round_data --n_windows 300

Model planners (controller/planner.py), one per checkpoint in --ckpts, named <kind>:<run>:
    mpc        replan every 0.2 s: warm the model on the last 10 real frames, imagine 6 constant-acceleration plans
               for 2 s (S prior samples x M dropout masks), apply the first action of the cheapest (cost.plan_costs)
    mpc_warp   the same with the predict-then-warp baseline (warp_plan.warp_imagine, coast reference)
               e.g. --policies human,coast --ckpts runs/ab_inv4_01/last.pt runs/ab_anchor/last.pt --planners mpc,mpc_warp
Learned policies (scripts/train_policy.py): --policy_ckpts runs/pol_v4/best.pt ..., named pol:<run>.
Per episode the planners also report the share of steps each plan was picked and the imagined risk of the picks.
Policies (constant / logged):
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
    print(f"  {'policy':26s} {'front':>6s} {'side':>6s} {'rear':>6s} {'gap<2m':>7s} {'minGap':>7s} {'hardBr':>7s} "
          f"{'progr':>6s} {'lag s':>6s} {'freeze':>7s}")
    for name, g in df.groupby("policy", sort=False):
        print(f"  {name:26s} {g.col_front.mean():6.3f} {g.col_side.mean():6.3f} {g.col_rear.mean():6.3f} "
              f"{(g.min_gap < 2).mean():7.3f} {g.min_gap.replace(np.inf, np.nan).median():7.1f} "
              f"{(g.hard_brakes > 0).mean():7.3f} {g.progress.median():6.2f} {g.lag_s.median():6.2f} "
              f"{(g.freeze_s >= 1).mean():7.3f}")


def load_planners(a, cfg):
    """{<kind>:<run>: ModelPlanner} for every checkpoint in --ckpts and kind in --planners."""
    if not a.ckpts:
        return {}
    import torch
    from roundabout_wm.controller.cost import CostConfig
    from roundabout_wm.controller.planner import ModelPlanner
    from roundabout_wm.models.world_model import WMConfig, WorldModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cc = CostConfig(w_prog=a.w_prog, **({} if a.w_epi is None else {"w_epi": a.w_epi}))
    print(f"model planners: {cc.to_dict()}")
    out = {}
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        for kind in a.planners.split(","):
            if kind not in ("mpc", "mpc_warp"):
                raise ValueError(f"unknown planner {kind}")
            out[f"{kind}:{run}"] = ModelPlanner(m, s["stats"], cfg, cc=cc, T=a.plan_T, samples=a.samples, masks=a.masks,
                                                seed=a.seed, warp_ref="coast" if kind == "mpc_warp" else None,
                                                name=f"{kind}:{run}", device=dev)
    return out


def load_policies(a, cfg):
    """{pol:<run>: LearnedPolicy} for every checkpoint in --policy_ckpts (each names its world model)."""
    if not a.policy_ckpts:
        return {}
    import torch
    from roundabout_wm.controller.policy import LearnedPolicy, PolicyNet
    from roundabout_wm.models.world_model import WMConfig, WorldModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for path in a.policy_ckpts:
        ps = torch.load(path, map_location=dev)
        s = torch.load(ps["wm"], map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        pi = PolicyNet(hidden=m.cfg.hidden).to(dev)
        pi.load_state_dict(ps["policy"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        out[f"pol:{run}"] = LearnedPolicy(m, pi, s["stats"], cfg, name=f"pol:{run}", device=dev)
    return out


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
    p.add_argument("--reactive", action="store_true", help="road users react to the ego (ReplayEnv(reactive=True))")
    p.add_argument("--ckpts", nargs="*", default=[], help="world models for the model planners")
    p.add_argument("--planners", default="mpc", help="comma list of planner kinds: mpc, mpc_warp")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--plan_T", type=int, default=10, help="planning horizon, steps (10 = 2 s)")
    p.add_argument("--w_prog", type=float, default=0.5,
                   help="planner progress weight (CostConfig default 0.2 makes coasting beat accelerating on an empty "
                        "road; 0.5 = Nov 13 calibration: the plan cheapest on real futures then never crashes)")
    p.add_argument("--w_epi", type=float, default=None, help="planner epistemic weight (default: CostConfig's)")
    p.add_argument("--policy_ckpts", nargs="*", default=[], help="learned policies (train_policy.py), named pol:<run>")
    a = p.parse_args()
    cfg = Config()
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, horizon=a.horizon, reactive=a.reactive)
    idx = np.sort(np.random.default_rng(a.seed).choice(len(env.ds), min(a.n_windows, len(env.ds)), replace=False))
    planners = {**load_planners(a, cfg), **load_policies(a, cfg)}
    names = [x for x in a.policies.split(",") if x] + list(planners)
    rows = []
    for j, i in enumerate(idx):
        for name in names:
            pol = planners[name] if name in planners else POLICIES[name]()
            row = dict(policy=name, **run_episode(env, pol, int(i)))
            if hasattr(pol, "episode_summary"):
                row.update(pol.episode_summary())
            rows.append(row)
        if (j + 1) % 25 == 0:
            print(f"  {j + 1}/{len(idx)} windows", end="\r")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    hum = df[df.policy == "human"].set_index("window")
    traffic = hum.index[hum.min_gap < a.near]
    summary(df, "all")
    pick_cols = [c for c in df if c.startswith("pick_")]
    if pick_cols:
        print("\nmodel planners: share of steps each plan was picked; imagined risk of the picks (mean)")
        print(df[df.policy.isin(list(planners))].groupby("policy", sort=False)[pick_cols + ["imagined_risk"]]
              .mean().round(3).to_string())
    summary(df[df.window.isin(traffic)], f"traffic: human had a road user < {a.near:g} m ahead")
    bad = hum[hum.col_front | hum.col_side]
    if len(bad):
        print(f"\nacceptance: the human collides (front/side) in {len(bad)} windows: {bad.index.tolist()[:20]}")
    else:
        print("\nacceptance: the human never collides front/side")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
