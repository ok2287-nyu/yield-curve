"""Controller diagnostics in the model (CPU), before any planner is trusted.

    python scripts/ctrl_diag.py --cache ../round_cache --ckpts runs/ab_anchor/last.pt runs/ab_fixedprior/last.pt \
        --n_windows 200 --samples 4 --masks 8

From the real warm-up of each validation window, five 2 s acceleration plans along the human's route:
    log (the human's own accelerations), brake (-3), gentle (-1.5), coast (0), go (+1.5 m/s^2)
are rolled out with S prior samples x M dropout masks (the same prior noise in every mask) and scored with
roundabout_wm/controller/cost.py.

Answers:
  1. Does the epistemic signal (std of the summed risk over dropout masks) rise for plans unlike the human's,
     above all hard braking, where the model's predictions are known to leak? If brake's epistemic is not clearly
     above log's, the dropout penalty will not protect a controller from exploiting leakage.
  2. Its distribution under the human's own plan: the 90th percentile is the suggested --epi_thresh (CostConfig).
  3. Which plan the model's cost prefers, with and without the epistemic term (no real futures are used here; the
     check against real futures needs the replay evaluation).
Windows with traffic near the ego ("contested": any plan's risk > --contested) are reported separately.
Writes ctrl_diag.csv (window, plan, risk, epistemic, progress, comfort, total) next to the first checkpoint.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig, plan_costs
from roundabout_wm.controller.imagine import ego_paths, imagine, route_from_item, warm_states
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import WMConfig, WorldModel

PLANS = {"brake": -3.0, "gentle": -1.5, "coast": 0.0, "go": 1.5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=200)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--contested", type=float, default=0.5, help="risk above which a window counts as contested")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, cc = Config(), CostConfig()
    T = a.horizon
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    rows = []
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=cfg.data.n_pred)
        idx = np.sort(np.random.default_rng(a.seed).choice(len(ds), min(a.n_windows, len(ds)), replace=False))
        names = ["log"] + list(PLANS)
        for j in range(0, len(idx), a.batch):
            items = [ds[int(i)] for i in idx[j:j + a.batch]]
            h, prev = warm_states(m, items, cfg, dev)
            for bi, it in enumerate(items):
                state0, rs, rk, log_acc = route_from_item(it, s["stats"], cfg)
                accels = np.stack([log_acc[:T]] + [np.full(T, v) for v in PLANS.values()])
                paths = ego_paths(state0, rs, rk, accels, cfg)
                out = imagine(m, h[bi:bi + 1], paths, s["stats"], cfg, a.samples, a.masks,
                              seed=a.seed + int(idx[j + bi]), prev0=prev[bi:bi + 1])
                c = plan_costs(out["green"], out["speed"], out["acc"], out["curv"], cfg.render, cc)
                for pi, name in enumerate(names):
                    rows.append(dict(run=run, window=int(idx[j + bi]), plan=name,
                                     **{k: float(c[k][pi]) for k in ("risk", "epistemic", "progress", "comfort")}))
            print(f"  {run}: {min(j + a.batch, len(idx))}/{len(idx)} windows", end="\r")
        print()
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "ctrl_diag.csv")
    df.to_csv(out, index=False)
    for run, d in df.groupby("run", sort=False):
        cont = d.groupby("window").risk.transform("max") > a.contested
        print(f"\n{run}: {d.window.nunique()} windows, {d[cont].window.nunique()} contested "
              f"(some plan's risk > {a.contested}); {a.samples} samples x {a.masks} masks, {T * cfg.data.dt:.0f} s")
        for label, sub in (("all", d), ("contested", d[cont])):
            if sub.empty:
                continue
            piv = sub.pivot(index="window", columns="plan")
            print(f"  [{label}] mean per plan (epistemic ratio = plan / log, per window, median):")
            print(f"  {'plan':7s} {'risk':>7s} {'epist':>7s} {'ratio':>6s} {'>log':>5s} {'progr':>6s} {'comf':>6s}")
            for name in ["log"] + list(PLANS):
                e = piv["epistemic"][name]; e0 = piv["epistemic"]["log"]
                ratio = np.median(e / np.maximum(e0, 1e-6))
                print(f"  {name:7s} {piv['risk'][name].mean():7.3f} {e.mean():7.3f} {ratio:6.2f} "
                      f"{(e > e0).mean():5.2f} {piv['progress'][name].mean():6.2f} {piv['comfort'][name].mean():6.2f}")
            e_log = piv["epistemic"]["log"]
            print(f"  epistemic under the human's plan: p50 {np.percentile(e_log, 50):.3f}  "
                  f"p90 {np.percentile(e_log, 90):.3f}  (suggested --epi_thresh = p90 over all windows)")
            for w_epi, thr in ((0.0, 0.0), (cc.w_epi, float(np.percentile(d[d.plan == 'log'].epistemic, 90)))):
                tot = (piv["risk"] + w_epi * np.maximum(piv["epistemic"] - thr, 0) + cc.w_prog * piv["progress"]
                       + piv["comfort"])
                pick = tot.idxmin(axis=1).value_counts(normalize=True)
                print(f"  cheapest plan by model cost (w_epi {w_epi}): " +
                      ", ".join(f"{k} {pick.get(k, 0):.2f}" for k in ["log"] + list(PLANS)))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
