"""How strongly does a lead's motion go with its follower's in the DATA? (tracks only, CPU, no model)

    python scripts/lead_slope.py --cache ../round_cache --data_dir ../round_data --split train

Why. A model trained by MSE on logs learns E[lead's future | past, ego's future actions]. If in the data the lead's
deviation from constant speed goes with the follower's (the ego's) with slope beta, controlling for what the model
can see, then a perfectly fitted conditional model moves the lead by beta * (ego's extra displacement) when the ego
changes plan: action gain = 1 - beta. So:
    measured gain (action_gain.py) ~ 1 - beta    -> the model learned the data's confounding (conditioning != do())
    measured gain << 1 - beta                    -> the model mirrors MORE than the data justify (a shortcut)
And the lead's own current acceleration as an extra control: if beta collapses, the confounder is mostly visible
state (the lead is already braking); showing the model the lead's acceleration would remove most of the leak. If
beta stays, the confounder is the lead's future decision: only invariance / factorisation can fix it.

Pairs. Every --every rows (1 s) of every car track ("follower" = the ego role), its lead = the nearest vehicle
ahead ON the follower's own future path (within --lat m of it, aligned within 45 degrees, bumper gap <= --max_gap m)
that is tracked from 0.4 s before to h s after. For each horizon h:
    dF = distance the follower travels in h  -  v_F(0) * h      (deviation from constant speed; < 0 = slowed)
    dL = the same for the lead
Regression  dL ~ beta * dF + c + gap + (v_L - v_F) + v_F + a_F   [+ a_L]
(a_F = the follower's own recent acceleration: the model sees the human's past actions). Fitted overall, by gap
and by the sign of dF; 95% intervals by resampling recordings.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from roundabout_wm.config import Config
from roundabout_wm.invariance import path_polyline
from roundabout_wm.kinematics import wrap_angle
from roundabout_wm.round_io import load_recording

LEAD_CLASSES = {"car", "truck_bus", "van", "trailer", "motorcycle"}
BASE = ["gap", "dv", "v_f", "a_f"]


def pairs_from_recording(rec, cfg, horizons=(1.0, 2.0), every=5, lat=2.5, max_gap=30.0):
    dt = cfg.data.dt
    H = {h: int(round(h / dt)) for h in horizons}
    Hmax = max(H.values())
    meta = rec.tracks_meta
    tr = {int(t): (g.frame.to_numpy(), g[["x", "y"]].to_numpy(), g.heading.to_numpy(), g.speed.to_numpy())
          for t, g in rec.tracks.groupby("trackId")}
    at = {int(f): g[["trackId", "x", "y", "heading"]].to_numpy() for f, g in rec.tracks.groupby("frame")}
    rows = []
    for fid, (fr, xy, hd, v) in tr.items():
        if meta.at[fid, "class"] not in cfg.data.ego_classes:
            continue
        Lf = float(meta.at[fid, "length"])
        for j in range(2, len(fr) - Hmax, every):
            pts, phd, arc = path_polyline(xy[j:], hd[j], length=max_gap + 15.0)
            best = None
            for tid, x, y, h in at[int(fr[j])]:
                tid = int(tid)
                if tid == fid or meta.at[tid, "class"] not in LEAD_CLASSES:
                    continue
                d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
                k = int(d.argmin())
                gap = arc[k] - Lf / 2 - float(meta.at[tid, "length"]) / 2
                if d[k] < lat and arc[k] > Lf / 2 and abs(wrap_angle(h - phd[k])) < np.pi / 4 and gap <= max_gap:
                    if best is None or arc[k] < best[1]:
                        best = (tid, arc[k], gap)
            if best is None:
                continue
            tid, _, gap = best
            lfr, lxy, _, lv = tr[tid]
            jl = int(np.searchsorted(lfr, fr[j]))
            if jl < 2 or jl + Hmax >= len(lfr) or lfr[jl + Hmax] != fr[j + Hmax] or lfr[jl - 2] != fr[j - 2]:
                continue                                              # the lead is not tracked over the window
            row = dict(rec=rec.rec_id, follower=fid, lead=tid, frame=int(fr[j]), gap=gap, v_f=v[j], v_l=lv[jl],
                       dv=lv[jl] - v[j], a_f=(v[j] - v[j - 2]) / (2 * dt), a_l=(lv[jl] - lv[jl - 2]) / (2 * dt))
            for h, n in H.items():
                df_ = np.linalg.norm(np.diff(xy[j:j + n + 1], axis=0), axis=1).sum() - v[j] * n * dt
                dl_ = np.linalg.norm(np.diff(lxy[jl:jl + n + 1], axis=0), axis=1).sum() - lv[jl] * n * dt
                row[f"dF_{h:g}"], row[f"dL_{h:g}"] = df_, dl_
            rows.append(row)
    return rows


def fit_beta(d, h, extra=()):
    """OLS dL_h ~ beta * dF_h + const + BASE + extra -> (beta, r2, n)."""
    if len(d) < len(BASE) + len(extra) + 3:
        return np.nan, np.nan, len(d)
    X = np.column_stack([d[f"dF_{h:g}"], np.ones(len(d))] + [d[c] for c in BASE + list(extra)])
    y = d[f"dL_{h:g}"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return float(coef[0]), float(r2), len(d)


def boot_ci(d, h, extra=(), B=500, seed=0):
    """95% interval of beta, resampling recordings."""
    recs = d.rec.unique()
    if len(recs) < 2:
        return np.nan, np.nan
    rng, groups = np.random.default_rng(seed), {r: g for r, g in d.groupby("rec")}
    bs = []
    for _ in range(B):
        s = pd.concat([groups[r] for r in rng.choice(recs, len(recs))])
        bs.append(fit_beta(s, h, extra)[0])
    return tuple(np.nanpercentile(bs, [2.5, 97.5]))


def report(df, horizons, B):
    for h in horizons:
        print(f"\n=== horizon {h:g} s  (implied action gain of a perfectly fitted conditional model = 1 - beta)")
        print(f"  {'subset':30s} {'n':>6s}  {'beta':>6s} [95% CI]        1-beta  | +a_L: beta [95% CI]        R2")
        sub = {"all": df, "moving (v_F > 1)": df[df.v_f > 1]}
        for lo, hi in ((0, 10), (10, 20), (20, 31)):
            sub[f"gap {lo}-{hi} m"] = df[(df.gap >= lo) & (df.gap < hi)]
        dF = df[f"dF_{h:g}"]
        sub["follower slowed (dF < -0.5)"] = df[dF < -0.5]
        sub["follower about const (|dF|<=0.5)"] = df[dF.abs() <= 0.5]
        sub["follower sped up (dF > 0.5)"] = df[dF > 0.5]
        for name, d in sub.items():
            b, r2, n = fit_beta(d, h)
            lo, hi = boot_ci(d, h, B=B)
            b2, r22, _ = fit_beta(d, h, ("a_l",))
            lo2, hi2 = boot_ci(d, h, ("a_l",), B=B)
            print(f"  {name:30s} {n:6d}  {b:6.2f} [{lo:5.2f}, {hi:5.2f}]  {1 - b:6.2f}  |  {b2:5.2f} [{lo2:5.2f}, "
                  f"{hi2:5.2f}]  {r2:.2f}/{r22:.2f}")
    print("\nCompare with action_gain.py (ab_anchor, lead): brake 0.52 at 1 s / 0.62 at 2 s, go 0.38 at 2 s. The plans"
          " move the ego by 1.5-6 m, often more than |dF| in the data: read the bins, and treat 1 - beta as the gain"
          " of a linear fit (the model may extrapolate differently).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="for split.json")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--horizons", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--every", type=int, default=5, help="rows between samples (5 = 1 s)")
    p.add_argument("--lat", type=float, default=2.5)
    p.add_argument("--max_gap", type=float, default=30.0)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--out", default="runs/lead_slope_pairs.csv")
    p.add_argument("--csv", default=None, help="skip the tracks and report from this pairs csv")
    a = p.parse_args()
    if a.csv:
        df = pd.read_csv(a.csv)
    else:
        cfg = Config()
        recs = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
        rows = []
        for r in recs:
            rows += pairs_from_recording(load_recording(a.data_dir, r, step=cfg.data.step), cfg, a.horizons,
                                         a.every, a.lat, a.max_gap)
            print(f"  rec {r}: {len(rows)} pairs so far")
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    report(df, a.horizons, a.boot)


if __name__ == "__main__":
    main()
