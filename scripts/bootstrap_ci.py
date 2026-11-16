"""95% intervals for the headline numbers, resampling RECORDINGS (windows of one recording are not independent).
CSV only, CPU, seconds.

    python scripts/bootstrap_ci.py --cache ../round_cache --gain runs/ab_anchor/action_gain.csv \
        --plan_rank runs/ab_anchor/plan_rank.csv

    action_gain.csv  median gain per run / group / plan / time, with its interval
    plan_rank.csv    per run, contested windows (some plan's real risk without followers > --contested):
                     the model pick's real front-collision rate (the "exploit rate") and the human's;
                     brake rated riskier than the human's plan: by the model vs really
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from plan_rank import model_pick
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.dataset import window_starts


def window_recordings(cache, split, cfg):
    """dataset window index -> recording id, as RounDWindows / ReplayEnv number the windows."""
    recs = json.load(open(os.path.join(cache, "split.json")))[split]
    out = []
    for r in recs:
        n = len(window_starts(os.path.join(cache, f"rec_{r:02d}"), cfg.data.n_cond + cfg.data.n_pred,
                              cfg.data.window_stride))
        out += [r] * n
    return np.array(out)


def cluster_boot(d, stat, B=1000, seed=0):
    """stat(rows) on resamples of whole recordings -> (2.5%, 97.5%)."""
    groups = {r: g for r, g in d.groupby("rec")}
    keys = list(groups)
    if len(keys) < 2:
        return np.nan, np.nan
    rng, vals = np.random.default_rng(seed), []
    for _ in range(B):
        vals.append(stat(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def gain_table(g, B):
    print("\n===== action gain: median [95% CI over recordings] (n)")
    for run, d in g.groupby("run", sort=False):
        print(f"--- {run} ({d.rec.nunique()} recordings)")
        for (grp, plan, t), s in d.groupby(["group", "plan", "t"]):
            if len(s) < 5:
                continue
            lo, hi = cluster_boot(s, lambda x: x.gain.median(), B)
            print(f"  {grp:10s} {plan:6s} {t:3.1f} s  {s.gain.median():5.2f} [{lo:5.2f}, {hi:5.2f}]  (n {len(s)})")


def plan_table(pr, cc, contested, B):
    print(f"\n===== plan ranking, contested windows (some plan's real* risk > {contested})")
    if "real_risk_nf" not in pr:
        pr = pr.assign(real_risk_nf=pr.real_risk)
    for run, d in pr.groupby("run", sort=False):
        cont = d.groupby("window").real_risk_nf.transform("max") > contested
        d = d[cont]

        def exploit(x):
            return model_pick(x, cc).front.mean()

        def human(x):
            return x[x.plan == "log"].front.mean()

        def brake_worse(x, kind):
            w = x.pivot_table(index="window", columns="plan", values=f"{kind}_risk")
            return (w["brake"] > w["log"]).mean()

        rows = [("model pick: real front collisions", exploit), ("human plan: real front collisions", human),
                ("model rates brake riskier than log", lambda x: brake_worse(x, "model")),
                ("really riskier (real* risk)", lambda x: brake_worse(x.assign(real_risk=x.real_risk_nf), "real"))]
        print(f"--- {run}: {d.window.nunique()} contested windows, {d.rec.nunique()} recordings")
        for name, f in rows:
            lo, hi = cluster_boot(d, f, B)
            print(f"  {name:38s} {f(d):5.3f} [{lo:5.3f}, {hi:5.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--gain", nargs="+", default=None, help="one or more action_gain csv files (concatenated)")
    p.add_argument("--plan_rank", nargs="+", default=None, help="one or more plan_rank csv files (concatenated)")
    p.add_argument("--contested", type=float, default=0.5)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    rec_of = window_recordings(a.cache, a.split, Config())
    if a.gain:
        g = pd.concat([pd.read_csv(f) for f in a.gain], ignore_index=True)
        g = g.assign(rec=rec_of[g.window.to_numpy()], run=g.run if "run" in g else "model")
        gain_table(g, a.boot)
    if a.plan_rank:
        pr = pd.concat([pd.read_csv(f) for f in a.plan_rank], ignore_index=True)
        pr = pr.assign(rec=rec_of[pr.window.to_numpy()], run=pr.run if "run" in pr else "model")
        plan_table(pr, CostConfig(), a.contested, a.boot)


if __name__ == "__main__":
    main()
