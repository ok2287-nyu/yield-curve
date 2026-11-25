"""Did the model know it was in trouble? Uncertainty in the steps before the planner's collisions.

    python scripts/crash_uncertainty.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_inv4_01/last.pt runs/ab_anchor/last.pt --n_windows 1000 --out runs/planner_eval/crash_unc.csv

Runs the model planner (controller/planner.py, mpc) closed loop in ReplayEnv and records, at every decision, the
uncertainty signals of the plan it picked:
    epi      std over dropout masks of the plan's summed risk (cost.plan_costs' epistemic term; the planner's penalty)
    epi_pix  PPUU's: pixel variance over dropout masks of the imagined frames, summed over pixels, mean over steps
    risk     the plan's imagined risk (mean over masks and samples)
plus the REAL gap to the car ahead at that moment (front_gap; > 40 m or none = 40).

Pre-crash steps: decisions within --before steps (default 5 = 1 s) before a front or side collision. Controls:
every decision in episodes without any collision. Prints, per planner:
    - each signal's mean before crashes vs in controls, and its AUROC (pre-crash vs control decisions);
    - the same AUROC with the controls MATCHED on the real gap (within gap bins, weighted like the pre-crash steps):
      does the signal add anything beyond "a car is close"?
    - the baseline "closeness" = −real gap, for comparison;
    - the picked plan's signals by steps before the hit (5 ... 1) vs the control mean.
A signal that could stop these crashes needs a high gap-matched AUROC. ~0.5 = no warning beyond the gap.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from roundabout_wm.config import Config
from roundabout_wm.controller.replay import ReplayEnv

SIGNALS = ("epi", "epi_pix", "risk", "closeness")
GAP_BINS = (0, 2, 4, 6, 10, 20, 41)


def auroc(pos, neg):
    from yield_starts import auroc as a
    return a(pos, neg)


def matched_auroc(pre, ctl, key):
    """AUROC within real-gap bins, averaged with the pre-crash steps' bin counts as weights."""
    num = den = 0.0
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        p = pre[(pre.gap >= lo) & (pre.gap < hi)][key]
        c = ctl[(ctl.gap >= lo) & (ctl.gap < hi)][key]
        if len(p) and len(c):
            num += len(p) * auroc(p, c)
            den += len(p)
    return num / den if den else float("nan")


def report(steps, eps, name, before):
    d = steps[steps.policy == name]
    e = eps[eps.policy == name]
    hit = e[e.col_front | e.col_side].copy()
    hit["t_hit"] = hit[["t_front", "t_side"]].min(axis=1)
    clean = e[~(e.col_front | e.col_side | e.col_rear)]
    dt = 0.2
    d = d.merge(hit[["window", "t_hit"]], on="window", how="left")
    d["to_hit"] = np.round(d.t_hit / dt) - d.t
    pre = d[(d.to_hit >= 1) & (d.to_hit <= before)]
    ctl = d[d.window.isin(clean.window)]
    print(f"\n== {name}: {len(e)} episodes, {len(hit)} with a front/side collision, {len(clean)} without any; "
          f"{len(pre)} decisions in the {before * dt:.1f} s before a hit, {len(ctl)} control decisions")
    if pre.empty or ctl.empty:
        return {}
    print(f"   {'signal':10s} {'pre-crash':>10s} {'control':>10s} {'AUROC':>6s} {'gap-matched':>12s}")
    out = {"policy": name, "n_hit": len(hit), "n_pre": len(pre)}
    for k in SIGNALS:
        a, am = auroc(pre[k], ctl[k]), matched_auroc(pre, ctl, k)
        print(f"   {k:10s} {pre[k].mean():10.4g} {ctl[k].mean():10.4g} {a:6.2f} {am:12.2f}")
        out[f"auroc_{k}"], out[f"auroc_matched_{k}"] = a, am
    print(f"   real gap before the hit: median {pre.gap.median():.1f} m (controls {ctl.gap.median():.1f} m)")
    print("   picked plan's epi by steps before the hit (control mean " + f"{ctl.epi.mean():.4g}): "
          + "  ".join(f"{int(k)}: {g.epi.mean():.4g} (n {len(g)})" for k, g in pre.groupby("to_hit", sort=True)
                      .__iter__()))
    print("   picks before the hit: " + ", ".join(f"{k} {v:.2f}" for k, v in pre.pick.value_counts(normalize=True)
                                                    .items()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--reactive", action="store_true")
    p.add_argument("--before", type=int, default=5, help="decisions before the hit counted as pre-crash (5 = 1 s)")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--w_prog", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="CSV of every decision (an _episodes.csv is written next to it)")
    a = p.parse_args()
    import replay_eval as rv
    cfg = Config()
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, horizon=a.horizon, reactive=a.reactive)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(env.ds), min(a.n_windows, len(env.ds)), replace=False)
    ns = argparse.Namespace(ckpts=a.ckpts, planners="mpc", samples=a.samples, masks=a.masks, plan_T=10,
                            w_prog=a.w_prog, w_epi=None, seed=a.seed, policy_ckpts=[])
    planners = rv.load_planners(ns, cfg)
    steps, eps = [], []
    for j, w in enumerate(idx):
        for name, pol in planners.items():
            pol.record_trace = True
            pol.trace = []
            obs = env.reset(int(w))
            pol.reset(env)
            while not env.done:
                obs = env.step(pol.act(env, obs))
            met = env.metrics()
            pol.episode_summary()
            gaps = {int(r["t"]): min(float(r["gap"]), 40.0) for r in env.log}
            for r in pol.trace:
                r.update(policy=name, gap=gaps.get(r["t"], 40.0), closeness=-gaps.get(r["t"], 40.0))
                steps.append(r)
            eps.append(dict(policy=name, **met))
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(idx)} windows", flush=True)
    steps, eps = pd.DataFrame(steps), pd.DataFrame(eps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    steps.to_csv(a.out, index=False)
    eps.to_csv(a.out.replace(".csv", "_episodes.csv"), index=False)
    for name in planners:
        report(steps, eps, name, a.before)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
