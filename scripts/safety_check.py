"""Two follow-ups to plan_rank / exploit_cases (CPU or GPU), on the same windows and five plans.

    python scripts/safety_check.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_anchor/last.pt runs/ab_fixedprior/last.pt --n_windows 300 --samples 4 --masks 4

A. Headway check (controller/cost.py headway_violation). The model almost never draws a collision the ego's own
   motion causes (it shrinks the car instead), so a collision cost on its pixels is blind. Instead: in each predicted
   frame (green averaged over samples and masks) the gap to the nearest pixel above --thresh in the ego's lane
   corridor; a plan is rejected if the gap ever drops below d0 + tau * v, optionally with persistence (a lead cannot
   pull away faster than lead_vmax, so a car that fades or vanishes while the ego closes in still counts).
   Per setting:
     recall        share of plans that really crash (first collision front or side) that the check rejects
     FA human      share of windows where it rejects the human's own plan (false alarm: the human never crashes)
     FA other      share of the other non-crashing plans it rejects
     pick crash    the model's pick among the plans that pass (all rejected -> brake): real crash rate
   "oracle" rows apply the same rule to the TRUE gaps (box geometry): the best any model could do with this rule.
B. Braking pessimism. The model rates braking riskier than the human's plan far more often than it is. Where does the
   extra predicted risk sit? In the proximity zone of each plan (weights = cost.proximity_mask):
     front mass    model green and real green (no followers) weighted by the zone, summed over the steps
     phantom       model mass in the zone where no real road user is (real footprints dilated 2 px)
     gap at 2 s    the predicted vs the real gap to the lead (inf -> 40 m)
   Brake vs log, in windows where neither crashes. Phantom up under brake = cars appear because the ego brakes
   (leakage); gap not opening under brake while the real one does = the lead "waits" for the ego.
Writes safety_check.csv next to the first checkpoint. Rendered rounD data: HPC / your own machine only.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from plan_rank import NAMES, model_pick, score_window
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig, headway_violation, predicted_gap, proximity_mask
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel

THRESH = (0.15, 0.3)
D0, TAU, LEAD = (1.0, 2.0), (0.5, 1.0), (None, 8.0)


def combos():
    for th, d0, tau, lv in itertools.product(THRESH, D0, TAU, LEAD):
        yield f"m_t{th:g}_d{d0:g}_tau{tau:g}_" + ("p" if lv else "np"), (th, d0, tau, lv)
    for d0, tau in itertools.product(D0, TAU):
        yield f"oracle_d{d0:g}_tau{tau:g}", (None, d0, tau, None)


def window_rows(rows, ctx, cfg, cc):
    """Add headway violations, first-hit class and the braking decomposition to the rows of one window."""
    rc, dt = cfg.render, cfg.data.dt
    real, paths = ctx["real"], ctx["paths"]
    G = ctx["green"].float().mean((0, 1)).cpu()                                 # (P, T, 64, 64)
    P, T = G.shape[:2]
    v = torch.as_tensor(paths["state"][:, 1:T + 1, 3], dtype=torch.float32)    # (P, T)
    R = torch.from_numpy(real["green_nf"][:, :T]).float() / 255.0
    gaps = {th: predicted_gap(G, rc, cc, thresh=th) for th in THRESH}
    true_gap = torch.as_tensor(real["gap"][:, :T], dtype=torch.float32)
    viol = {}
    for name, (th, d0, tau, lv) in combos():
        g = true_gap if th is None else gaps[th]
        viol[name] = headway_violation(g, v, d0, tau, lv, dt)[0].numpy()
    W = proximity_mask(v, rc, cc)                                                # (P, T, 64, 64)
    occ = F.max_pool2d((R > 0.1).float().flatten(0, 1)[:, None], 5, 1, 2)[:, 0].view_as(R)
    front_m, front_r = (G * W).sum((-1, -2, -3)), (R * W).sum((-1, -2, -3))
    phantom = (G * W * (1 - occ)).sum((-1, -2, -3))
    gap_m_end = torch.clamp(gaps[0.3][:, -1], max=40.0)
    gap_r_end = torch.clamp(true_gap[:, -1], max=40.0)
    for p, r in enumerate(rows):
        first = next((c for h in real["hit_ids"][p] for _, c in h), None)
        r.update(crash=first in ("front", "side"), first_hit=first or "", front_mass_model=float(front_m[p]),
                 front_mass_real=float(front_r[p]), phantom=float(phantom[p]), gap_end_model=float(gap_m_end[p]),
                 gap_end_real=float(gap_r_end[p]), **{k: bool(x[p]) for k, x in viol.items()})
    return rows


def report_headway(d, cc):
    others = d[d.plan != "log"]
    crash = others[others.crash]
    print(f"  A. headway check: {len(crash)} crashing plans (first hit front/side) in {crash.window.nunique()} "
          f"windows; {others.window.nunique()} windows")
    print(f"    {'setting':28s} {'recall':>7s} {'FA human':>9s} {'FA other':>9s} {'pick crash':>11s} "
          f"{'human kept':>11s}")
    base = model_pick(d, cc)
    print(f"    {'no check':28s} {'':>7s} {'':>9s} {'':>9s} {base.crash.mean():11.3f} {'':>11s}")
    res = []
    for name, _ in combos():
        rec = crash[name].mean() if len(crash) else np.nan
        fa_h = d[d.plan == "log"][name].mean()
        fa_o = others[~others.crash][name].mean()
        ok = d[~d[name]]
        pick = model_pick(ok, cc) if len(ok) else ok
        missing = sorted(set(d.window) - set(pick.window))
        fallback = d[d.window.isin(missing) & (d.plan == "brake")]
        pick = pd.concat([pick, fallback])
        res.append((name, rec, fa_h, fa_o, pick.crash.mean(), (pick.plan == "log").mean()))
    for r in sorted(res, key=lambda r: (r[0].startswith("oracle"), -(r[1] - r[2]))):
        print(f"    {r[0]:28s} {r[1]:7.2f} {r[2]:9.2f} {r[3]:9.2f} {r[4]:11.3f} {r[5]:11.2f}")


def report_braking(d):
    piv = d.pivot(index="window", columns="plan")
    ok = ~(piv.crash["brake"] | piv.crash["log"] | (piv.first_hit["brake"] == "rear"))
    piv = piv[ok]
    print(f"\n  B. braking vs the human's plan ({len(piv)} windows where neither collides)")
    print(f"    {'plan':7s} {'model risk':>10s} {'real* risk':>10s} {'front mass M':>13s} {'front mass R':>13s} "
          f"{'phantom':>8s} {'gap 2s M':>9s} {'gap 2s R':>9s}")
    for name in ("log", "gentle", "brake", "coast", "go"):
        print(f"    {name:7s} {piv.model_risk[name].mean():10.3f} {piv.real_risk_nf[name].mean():10.3f} "
              f"{piv.front_mass_model[name].mean():13.2f} {piv.front_mass_real[name].mean():13.2f} "
              f"{piv.phantom[name].mean():8.2f} {piv.gap_end_model[name].median():9.1f} "
              f"{piv.gap_end_real[name].median():9.1f}")
    dm = piv.model_risk["brake"] - piv.model_risk["log"]
    worse = dm > 0
    dph = (piv.phantom["brake"] - piv.phantom["log"])[worse]
    dfm = (piv.front_mass_model["brake"] - piv.front_mass_model["log"])[worse]
    dfr = (piv.front_mass_real["brake"] - piv.front_mass_real["log"])[worse]
    dgm = (piv.gap_end_model["brake"] - piv.gap_end_model["log"])[worse]
    dgr = (piv.gap_end_real["brake"] - piv.gap_end_real["log"])[worse]
    print(f"    windows where the model rates brake riskier than log: {worse.mean():.2f}; in those, brake - log:")
    print(f"      front mass model {dfm.mean():+.2f} vs real {dfr.mean():+.2f}; phantom {dph.mean():+.2f} "
          f"(share of the model's extra front mass that is phantom: "
          f"{(dph.clip(lower=0).sum() / max(dfm.clip(lower=0).sum(), 1e-6)):.2f})")
    print(f"      gap to the lead at 2 s (median): model {dgm.median():+.1f} m vs real {dgr.median():+.1f} m")


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="only report from this safety_check.csv")
    a = p.parse_args()
    cfg, cc = Config(), CostConfig()
    if a.csv:
        df = pd.read_csv(a.csv)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
        rows, envs, real_cache = [], {}, {}
        for path in a.ckpts:
            s = torch.load(path, map_location=dev)
            m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
            m.load_state_dict(s["model"])
            run = os.path.basename(os.path.dirname(os.path.abspath(path)))
            key = json.dumps(s["stats"], sort_keys=True)
            if key not in envs:
                envs[key] = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=a.horizon)
            env = envs[key]
            idx = np.sort(np.random.default_rng(a.seed).choice(len(env.ds), min(a.n_windows, len(env.ds)),
                                                               replace=False))
            for j, i in enumerate(idx):
                r = score_window(m, s, env, int(i), a.horizon, a.samples, a.masks, a.seed, cc, cfg, dev,
                                 real_cache, run)
                if r is not None:
                    rows.extend(window_rows(r[0], r[1], cfg, cc))
                if (j + 1) % 25 == 0:
                    print(f"  {run}: {j + 1}/{len(idx)} windows", end="\r")
            print()
        df = pd.DataFrame(rows)
        out = os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "safety_check.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    for run, d in df.groupby("run", sort=False):
        print(f"\n===== {run}: {d.window.nunique()} windows, {a.horizon * cfg.data.dt:.0f} s plans")
        report_headway(d, cc)
        report_braking(d)


if __name__ == "__main__":
    main()
