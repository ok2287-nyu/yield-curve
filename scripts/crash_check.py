"""In the crash windows: did the model MOVE the hit car (mirroring) or ERASE it (it cannot draw a car at the ego's
bumper)? Both give a low or negative action gain, because the gain uses the centre of the predicted car. (CPU ok)

    python scripts/crash_check.py --cache ../round_cache --data_dir ../round_data --ckpt runs/ab_anchor/last.pt \
        --exploit runs/ab_anchor/exploit_cases.csv --gain runs/ab_anchor/action_gain.csv --n_controls 60

For every crashing plan in exploit_cases.csv (same run) and for --n_controls non-crash windows with a lead (from
action_gain.csv), at 1 s, 2 s and the hit step, for every plan, the hit car (the lead for controls):
    gain     model shift / true shift vs the human's plan, measured with the centre of the predicted green in a WIDE
             region around the car (its true footprints under all plans, grown 8 px along the ego's forward axis and
             2 px sideways, minus pixels near other road users), so a car the model displaces is still found
    mass_w   predicted green in that wide region / the car's true area      (did the car keep its pixels?)
    mass_n   the same in the narrow region (true footprints grown 1 px)
Reading, for the crash plan (mostly go) at the hit step:
    mass_w(plan) / mass_w(human) ~ 1 and gain low/negative  -> the car was MOVED (mirroring)
    mass_w(plan) / mass_w(human) << 1                       -> the car was ERASED near the ego (unfamiliar state)
and the brake gain in the same windows: normal (as in controls) -> the problem is specific to the crash plan.
Writes crash_check.csv next to the checkpoint.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pandas as pd
import torch

from plan_rank import NAMES, score_window
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel
from roundabout_wm.render import render_frame, world_to_ego_px

K3, K5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
KW = np.ones((17, 5), np.uint8)                                       # 8 px along rows (forward), 2 px across


def measure(env, ctx, tid, steps, rc):
    """-> rows for road user tid at the given step indices (k: +(k + 1) * 0.2 s), every plan."""
    poses = ctx["real"]["poses"]
    G = ctx["green"].float().mean((0, 1)).cpu().numpy()               # (P, T, H, W)
    rr, cc = np.mgrid[0:rc.size, 0:rc.size]
    rows = []
    for k in steps:
        if k >= G.shape[1]:
            continue
        o = env.R.others(env.frame_at(k + 1), env.ego_id)
        m = o["trackId"] == tid
        if not m.any():
            continue
        sel = {kk: v[m] for kk, v in o.items()}
        rest = {kk: v[~m] for kk, v in o.items()}
        narrow = np.zeros((rc.size, rc.size), bool)
        wide, near = narrow.copy(), narrow.copy()
        fps, true_c, edge = [], [], []
        for p in range(len(NAMES)):
            ego = env.ego_dict(tuple(poses[p, k + 1]))
            fp = (render_frame(ego, sel, None, rc)[1] > 25).astype(np.uint8)
            fps.append(fp)
            narrow |= cv2.dilate(fp, K3) > 0
            wide |= cv2.dilate(fp, KW) > 0
            near |= cv2.dilate((render_frame(ego, rest, None, rc)[1] > 25).astype(np.uint8), K5) > 0
            c = world_to_ego_px(np.array([sel["x"][0], sel["y"][0]]), ego["x"], ego["y"], ego["heading"], rc)
            true_c.append(np.array([c[1], c[0]]))
            edge.append(bool(fp[[0, -1]].any() or fp[:, [0, -1]].any()))
        narrow &= ~near
        wide &= ~near
        area = max(int(fps[0].sum()), 1)
        model_c, mass_w, mass_n = [], [], []
        for p in range(len(NAMES)):
            w = G[p, k] * wide
            mass_w.append(float(w.sum()) / area)
            mass_n.append(float((G[p, k] * narrow).sum()) / area)
            model_c.append(np.array([(w * rr).sum(), (w * cc).sum()]) / w.sum() if w.sum() > 1e-6 else None)
        for p in range(len(NAMES)):
            dt_ = true_c[p] - true_c[0]
            n2 = float(dt_ @ dt_)
            gain = np.nan
            if p > 0 and n2 >= 1.0 and model_c[p] is not None and model_c[0] is not None:
                gain = float((model_c[p] - model_c[0]) @ dt_) / n2
            rows.append(dict(t=round((k + 1) * 0.2, 1), plan=NAMES[p], gain=gain, true_shift_px=np.sqrt(n2),
                             mass_w=mass_w[p], mass_n=mass_n[p], mass_w_ratio=mass_w[p] / max(mass_w[0], 1e-6),
                             edge=edge[p]))
    return rows


def summary(df):
    for kind, d in df.groupby("kind", sort=False):
        nw = d.window.nunique()
        print(f"\n=== {kind}: {nw} windows")
        print(f"  {'plan':7s} {'t':>4s}   gain median [IQR]        mass_w ratio median   share < 0.5   n")
        for plan in ("brake", "gentle", "go"):
            for t in sorted(d.t.unique()):
                s = d[(d.plan == plan) & (d.t == t) & ~d.edge]
                if not len(s):
                    continue
                g = s.gain.dropna()
                q = g.quantile([0.25, 0.5, 0.75]).to_numpy() if len(g) else [np.nan] * 3
                print(f"  {plan:7s} {t:4.1f}   {q[1]:6.2f} [{q[0]:5.2f}, {q[2]:5.2f}]      {s.mass_w_ratio.median():6.2f}"
                      f"              {(s.mass_w_ratio < 0.5).mean():5.2f}      {len(s)}")
        hit = d[d.at_hit & (d.plan == d.crash_plan)]
        if len(hit):
            print(f"  at the hit step, crash plan: mass_w ratio median {hit.mass_w_ratio.median():.2f} "
                  f"(share < 0.5: {(hit.mass_w_ratio < 0.5).mean():.2f}), gain median {hit.gain.median():.2f}, "
                  f"n {len(hit)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--exploit", required=True, help="exploit_cases.csv")
    p.add_argument("--gain", default=None, help="action_gain.csv, to choose control windows with a lead")
    p.add_argument("--n_controls", type=int, default=60)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="skip the rollouts and summarise this crash_check.csv")
    a = p.parse_args()
    if a.csv:
        summary(pd.read_csv(a.csv)); return
    cfg, cc = Config(), CostConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(a.ckpt, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    e = pd.read_csv(a.exploit)
    e = e[e.run == run] if "run" in e else e
    cases = [("crash", int(r.window), int(r.track), r.plan, float(r.t_hit)) for r in e.itertuples()]
    if a.gain and a.n_controls > 0:
        g = pd.read_csv(a.gain)
        g = g[g.run == run] if "run" in g else g
        g = g[(g.group == "lead") & (g.plan == "go") & np.isclose(g.t, 2.0) & ~g.window.isin(e.window)]
        g = g.drop_duplicates("window")
        g = g.sample(min(a.n_controls, len(g)), random_state=a.seed)
        cases += [("control", int(r.window), int(r.track), "go", None) for r in g.itertuples()]
    split = json.load(open(os.path.join(a.cache, "split.json")))["val"]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=10)
    rows = []
    for i, (kind, w, tid, plan, t_hit) in enumerate(cases):
        r = score_window(m, s, env, w, 10, a.samples, a.masks, a.seed, cc, cfg, dev)
        if r is None:
            continue
        k_hit = int(round(t_hit / cfg.data.dt)) - 1 if t_hit else None
        steps = sorted({4, 9} | ({k_hit} if k_hit is not None else set()))
        for row in measure(env, r[1], tid, steps, cfg.render):
            rows.append(dict(kind=kind, window=w, track=tid, crash_plan=plan,
                             at_hit=k_hit is not None and abs(row["t"] - (k_hit + 1) * 0.2) < 1e-6, **row))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(cases)} cases")
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "crash_check.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    summary(df)


if __name__ == "__main__":
    main()
