"""Step 2: does the world model rank plans the way the real (replayed) future does? (CPU or GPU)

    python scripts/plan_rank.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_anchor/last.pt runs/ab_fixedprior/last.pt --n_windows 300 --samples 4 --masks 4

From the real warm-up of each validation window, the five 2 s plans of ctrl_diag.py (log = the human's own
accelerations, brake -3, gentle -1.5, coast 0, go +1.5) along the human's path are scored twice with the SAME cost
(controller/cost.py):
    model  the green channel imagined by the world model (S prior samples x M dropout masks)
    real   the green channel rendered at the plan's real poses among the logged road users (log replay: they do not
           react, so a plan that brakes is not rewarded by a follower that would have braked too)
plus a box-overlap check of every plan in the replay (front = the ego drove into someone).

Reports per checkpoint, for all windows and for contested ones (some plan's REAL risk > --contested):
  1. bias       mean model risk - real risk per plan. Brake's model risk above the real one = leakage / pessimism
                about braking, the effect that makes a controller unwilling to yield.
  2. ranking    per window, Spearman correlation of model and real risk over the five plans; how often the plan the
                model finds safest is the really safest.
  3. exploit    the plan the model's total cost picks: its REAL risk and real front-collision rate, against the
                human's plan. If the model's pick crashes more than the human, a controller will exploit the model.
  4. calibrate  with the real futures, how often the human's plan is the cheapest, for a small grid of progress /
                comfort weights, with acc_prev = the human's real previous acceleration (the old acc_prev = 0
                charged every plan a phantom jerk at the first step).
Writes plan_rank.csv (window, plan: model and real risk, epistemic, collisions, unweighted progress, acc^2, jerk^2)
next to the first checkpoint, so other weightings can be tried offline.
Rendered rounD data: HPC / your own machine only.
"""
import argparse
import itertools
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
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel

PLANS = {"brake": -3.0, "gentle": -1.5, "coast": 0.0, "go": 1.5}
NAMES = ["log"] + list(PLANS)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def total(d, cc, w_prog=None, w_acc=None, w_jerk=None, kind="real"):
    w_prog = cc.w_prog if w_prog is None else w_prog
    w_acc = cc.w_acc if w_acc is None else w_acc
    w_jerk = cc.w_jerk if w_jerk is None else w_jerk
    return d[f"{kind}_risk"] + w_prog * d.progress + w_acc * d.acc2 + w_jerk * d.jerk2


def report(df, cc, contested):
    for run, d in df.groupby("run", sort=False):
        cont = d.groupby("window").real_risk.transform("max") > contested
        print(f"\n===== {run}: {d.window.nunique()} windows, {d[cont].window.nunique()} contested "
              f"(some plan's real risk > {contested})")
        for label, sub in (("all", d), ("contested", d[cont])):
            if sub.empty:
                continue
            piv = sub.pivot(index="window", columns="plan")
            print(f"\n  [{label}] 1. risk per plan (mean)            3. exploit check")
            print(f"  {'plan':7s} {'model':>7s} {'real':>7s} {'bias':>7s} {'front':>6s} {'rear':>6s}")
            for name in NAMES:
                print(f"  {name:7s} {piv.model_risk[name].mean():7.3f} {piv.real_risk[name].mean():7.3f} "
                      f"{(piv.model_risk[name] - piv.real_risk[name]).mean():+7.3f} "
                      f"{piv.front[name].mean():6.3f} {piv.rear[name].mean():6.3f}")
            rho = [spearman(piv.model_risk.loc[w, NAMES], piv.real_risk.loc[w, NAMES]) for w in piv.index]
            same = (piv.model_risk[NAMES].idxmin(axis=1) == piv.real_risk[NAMES].idxmin(axis=1)).mean()
            print(f"  2. ranking: Spearman(model, real) over the 5 plans: median {np.nanmedian(rho):.2f}, "
                  f"mean {np.nanmean(rho):.2f} ({np.isnan(rho).sum()} windows tied); safest plan agrees {same:.2f}")
            for name in ("brake", "go"):
                dm = piv.model_risk[name] - piv.model_risk["log"]
                dr = piv.real_risk[name] - piv.real_risk["log"]
                big = dr.abs() > 0.1
                print(f"     {name} vs log: model says riskier {(dm > 0).mean():.2f}, really riskier {(dr > 0).mean():.2f}"
                      f", sign agrees {(np.sign(dm) == np.sign(dr))[big].mean():.2f} where |real diff| > 0.1 "
                      f"({big.sum()} windows)")
            sub2 = sub.assign(model_total=total(sub, cc, kind="model") + cc.w_epi * np.maximum(
                sub.model_epi - cc.epi_thresh, 0))
            pick = sub2.loc[sub2.groupby("window").model_total.idxmin()]
            log = sub[sub.plan == "log"]
            print(f"  3. the model's pick (weights as in CostConfig): " +
                  ", ".join(f"{k} {v:.2f}" for k, v in pick.plan.value_counts(normalize=True).items()))
            print(f"     its real risk {pick.real_risk.mean():.3f} vs the human's {log.real_risk.mean():.3f}; real "
                  f"front collisions {pick.front.mean():.3f} vs {log.front.mean():.3f}; rear {pick.rear.mean():.3f} "
                  f"vs {log.rear.mean():.3f}")
            print(f"  4. calibration on real futures: share of windows where the human's plan is the cheapest")
            res = []
            for wp, wa, wj in itertools.product((0.05, 0.1, 0.2, 0.5, 1.0), (0.0, 0.02, 0.1), (0.0, 0.001, 0.01)):
                t = sub.assign(tot=total(sub, cc, wp, wa, wj))
                best = t.loc[t.groupby("window").tot.idxmin()]
                res.append((wp, wa, wj, (best.plan == "log").mean(), best.front.mean()))
            res.sort(key=lambda r: -r[3])
            cur = [r for r in res if r[:3] == (cc.w_prog, cc.w_acc, cc.w_jerk)]
            for r in res[:5] + cur:
                tag = "  <- current" if r in cur else ""
                print(f"     w_prog {r[0]:<5g} w_acc {r[1]:<5g} w_jerk {r[2]:<6g} human cheapest {r[3]:.2f}, "
                      f"real front collisions of the real-cheapest plan {r[4]:.3f}{tag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=300)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--contested", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="skip the rollouts and only report from this plan_rank.csv")
    a = p.parse_args()
    cfg, cc = Config(), CostConfig()
    if a.csv:
        report(pd.read_csv(a.csv), cc, a.contested); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, dt, n = a.horizon, cfg.data.dt, cfg.data.n_cond
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    rows, envs, real_cache = [], {}, {}
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        key = json.dumps(s["stats"], sort_keys=True)
        if key not in envs:
            envs[key] = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=T)
        env = envs[key]
        idx = np.sort(np.random.default_rng(a.seed).choice(len(env.ds), min(a.n_windows, len(env.ds)),
                                                           replace=False))
        for j, i in enumerate(idx):
            i = int(i)
            env.reset(i)
            it = env.item
            state0, rs, rk, log_acc = route_from_item(it, s["stats"], cfg)
            accels = np.stack([log_acc[:T]] + [np.full(T, v) for v in PLANS.values()])
            paths = ego_paths(state0, rs, rk, accels, cfg)
            Tr = min(T, env.T)
            if Tr < T:
                continue                                       # the recording ends inside the horizon
            if i not in real_cache:
                real_cache[i] = env.open_loop(paths["state"][:, :, 3])
            real = real_cache[i]
            h, prev = warm_states(m, [it], cfg, dev)
            out = imagine(m, h, paths, s["stats"], cfg, a.samples, a.masks, seed=a.seed + i, prev0=prev)
            raw_prev = float(it["actions"][n - 2, 0] * s["stats"]["action_std"][0] + s["stats"]["action_mean"][0])
            cm = plan_costs(out["green"], out["speed"], out["acc"], out["curv"], cfg.render, cc, acc_prev=raw_prev)
            g_real = torch.from_numpy(real["green"]).float().div(255.0)[None, None]
            cr = plan_costs(g_real, out["speed"], out["acc"], out["curv"], cfg.render, cc, acc_prev=raw_prev)
            acc = paths["acc"]
            da = np.diff(np.concatenate([np.full((len(acc), 1), raw_prev), acc], 1), axis=1) / dt
            for pi, name in enumerate(NAMES):
                rows.append(dict(
                    run=run, window=i, plan=name,
                    model_risk=float(cm["risk"][pi]), model_epi=float(cm["epistemic"][pi]),
                    model_col=float(cm["collision"][..., pi, :].mean((0, 1)).sum()),
                    real_risk=float(cr["risk"][pi]), real_col=float(cr["collision"][0, 0, pi].sum()),
                    front=bool(real["front"][pi].any()), side=bool(real["side"][pi].any()),
                    rear=bool(real["rear"][pi].any()), min_gap=float(real["gap"][pi].min()),
                    progress=float(cm["progress"][pi]), acc2=float((acc[pi] ** 2).sum()),
                    jerk2=float((da[pi] ** 2).sum()), acc_prev=raw_prev))
            if (j + 1) % 25 == 0:
                print(f"  {run}: {j + 1}/{len(idx)} windows", end="\r")
        print()
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "plan_rank.csv")
    df.to_csv(out, index=False)
    report(df, cc, a.contested)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
