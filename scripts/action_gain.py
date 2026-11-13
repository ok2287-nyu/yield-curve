"""Does the model move each road user correctly when the ego changes its action? (CPU or GPU)

    python scripts/action_gain.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_anchor/last.pt runs/ab_fixedprior/last.pt --n_windows 300 --samples 4 --masks 4

Same windows and five plans as plan_rank.py. Under a different ego plan, every logged road user ends up at a
different place IN THE EGO'S FRAME, only because the ego is somewhere else (replay: nobody reacts). For each road
user and each plan q, at 1 s and 2 s:
    d_true  = its true position (centre, ego pixels) under plan q  -  under the human's plan
    d_model = the model's position (centroid of predicted green near it) under q  -  under the human's plan
    gain    = (d_model . d_true) / |d_true|^2      (only where |d_true| >= --min_shift px)
gain 1: the model shifts the road user exactly as the ego's action requires. gain 0: the road user stays glued to the
ego in the picture, as if it copied the ego's acceleration. Between: partly.
Groups (fixed at the start of the window, from the logged scene):
    lead          the nearest road user ahead in the ego's lane corridor (within 30 m)
    stationary    other road users that stand still for the whole 2 s (< 0.5 m/s): they cannot react to anything,
                  so their gain tests only whether the model applies the ego's own motion
    moving        all other road users
Reading: lead ~0 with stationary and moving ~1 -> the model makes the lead mirror the ego (car-following leakage: an
invented reaction). All ~0 -> the model under-applies the ego's own motion everywhere (action insensitivity).
The model's position: green (mean over samples x masks) inside the union of the road user's footprints under all
five plans, dilated by 3 px, minus pixels near any other road user; skipped if the model's mass there is below
--min_mass of the footprint area under the human's plan (the car has faded), or if the road user touches the image
edge under any plan. --sanity real / glued checks the measurement itself (expect 1 / 0).
Writes action_gain.csv next to the first checkpoint. Rendered rounD data: HPC / your own machine only.
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
from roundabout_wm.controller.replay import CORRIDOR, ReplayEnv, to_ego
from roundabout_wm.render import render_frame, world_to_ego_px

STEPS = (4, 9)                      # step index k <-> +(k + 1) * 0.2 s: 1 s and 2 s
K3 = np.ones((7, 7), np.uint8)      # dilation by 3 px
K2 = np.ones((5, 5), np.uint8)


def groups(env, n_steps):
    """{trackId: 'lead' | 'stationary' | 'moving'} for the road users present at the start of the episode."""
    o = env.R.others(env.frame_at(0), env.ego_id)
    x, y, h = env.pose
    L, W = env.ego_size
    out, best = {}, (np.inf, None)
    for j, tid in enumerate(o["trackId"]):
        tid = int(tid)
        fl = to_ego(np.array([o["x"][j], o["y"][j]]), x, y, h)
        if 0 < fl[0] < 30 + L / 2 and abs(fl[1]) < W / 2 + CORRIDOR + o["width"][j] / 2 and fl[0] < best[0]:
            best = (fl[0], tid)
        tr = env.R.tracks[tid]
        sp = tr[(tr.frame >= env.frame_at(0)) & (tr.frame <= env.frame_at(n_steps))].speed
        out[tid] = "stationary" if len(sp) and sp.abs().max() < 0.5 else "moving"
    if best[1] is not None:
        out[best[1]] = "lead"
    return out


def cases(env, ctx, rc, min_shift, min_mass, sanity=None):
    """sanity: None (the model), 'real' (the true frames as the "prediction": gain must be 1) or 'glued' (the human
    plan's true frames for every plan: gain must be 0). Checks the measurement itself."""
    poses = ctx["real"]["poses"]                                      # (P, T + 1, 3)
    if sanity == "real":
        G = ctx["real"]["green"].astype(np.float32) / 255
    elif sanity == "glued":
        G = np.repeat(ctx["real"]["green"][:1].astype(np.float32) / 255, len(NAMES), 0)
    else:
        G = ctx["green"].float().mean((0, 1)).cpu().numpy()           # (P, T, H, W)
    T = G.shape[1]
    grp = groups(env, T)
    rows = []
    rr, cc = np.mgrid[0:rc.size, 0:rc.size]
    for k in STEPS:
        if k >= T:
            continue
        o = env.R.others(env.frame_at(k + 1), env.ego_id)
        for tid, g in grp.items():
            m = o["trackId"] == tid
            if not m.any():
                continue                                              # left the recording
            sel = {kk: v[m] for kk, v in o.items()}
            rest = {kk: v[~m] for kk, v in o.items()}
            region = np.zeros((rc.size, rc.size), bool)
            near = np.zeros((rc.size, rc.size), bool)
            true_c, fps = [], []
            for p in range(len(NAMES)):
                ego = env.ego_dict(tuple(poses[p, k + 1]))
                fp = render_frame(ego, sel, None, rc)[1] > 25
                fps.append(fp)
                region |= cv2.dilate(fp.astype(np.uint8), K3) > 0
                near |= cv2.dilate((render_frame(ego, rest, None, rc)[1] > 25).astype(np.uint8), K2) > 0
                c = world_to_ego_px(np.array([sel["x"][0], sel["y"][0]]), ego["x"], ego["y"], ego["heading"], rc)
                true_c.append((c[1], c[0]))                           # (row, col)
            region &= ~near
            area = fps[0].sum()
            edge = any(f[[0, -1]].any() or f[:, [0, -1]].any() for f in fps)
            if area < 3 or region.sum() == 0 or edge:
                continue                                              # out of view or cut by the image edge
            model_c, ok = [], True
            for p in range(len(NAMES)):
                wgt = G[p, k] * region
                if wgt.sum() < min_mass * area:
                    ok = False; break
                model_c.append(((wgt * rr).sum() / wgt.sum(), (wgt * cc).sum() / wgt.sum()))
            if not ok:
                continue
            t0, m0 = np.array(true_c[0]), np.array(model_c[0])
            for p in range(1, len(NAMES)):
                dt_ = np.array(true_c[p]) - t0
                dm = np.array(model_c[p]) - m0
                n2 = float(dt_ @ dt_)
                if n2 < min_shift ** 2:
                    continue
                rows.append(dict(window=env.window, track=tid, group=g, plan=NAMES[p], t=(k + 1) * 0.2,
                                 shift_true=np.sqrt(n2), shift_model=float(np.linalg.norm(dm)),
                                 gain=float(dm @ dt_) / n2))
    return rows


def report(df):
    for run, d in df.groupby("run", sort=False):
        print(f"\n===== {run}: gain = how far the model moves a road user / how far it really moves in the ego's "
              f"frame, between plans (1 = right, 0 = glued to the ego)")
        for t, dt_ in d.groupby("t"):
            print(f"\n  at +{t:.0f} s   median gain [IQR] (n road users x plans)")
            print(f"  {'plan':7s} " + " ".join(f"{g:>26s}" for g in ("lead", "stationary", "moving")))
            for plan in NAMES[1:]:
                cells = []
                for g in ("lead", "stationary", "moving"):
                    x = dt_[(dt_.plan == plan) & (dt_.group == g)].gain
                    cells.append(f"{x.median():6.2f} [{x.quantile(.25):5.2f},{x.quantile(.75):5.2f}] "
                                 f"(n {len(x):4d})" if len(x) else f"{'-':>26s}")
                print(f"  {plan:7s} " + " ".join(cells))
            x = dt_.groupby("group").shift_true.median()
            print("  median true shift (px): " + ", ".join(f"{k} {v:.1f}" for k, v in x.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=300)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--min_shift", type=float, default=2.0, help="px; plans that move a road user less are skipped")
    p.add_argument("--min_mass", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="only report from this action_gain.csv")
    p.add_argument("--sanity", choices=["real", "glued"], default=None,
                   help="check the measurement: true frames as the prediction (gain 1) or glued to the human's plan (0)")
    a = p.parse_args()
    if a.csv:
        report(pd.read_csv(a.csv)); return
    cfg, cc = Config(), CostConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    rows, envs, real_cache = [], {}, {}
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        from roundabout_wm.models.world_model import WMConfig, WorldModel
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path))) + (f"_sanity_{a.sanity}" if a.sanity else "")
        key = json.dumps(s["stats"], sort_keys=True)
        if key not in envs:
            envs[key] = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=10)
        env = envs[key]
        idx = np.sort(np.random.default_rng(a.seed).choice(len(env.ds), min(a.n_windows, len(env.ds)),
                                                           replace=False))
        for j, i in enumerate(idx):
            r = score_window(m, s, env, int(i), 10, a.samples, a.masks, a.seed, cc, cfg, dev, real_cache, run)
            if r is not None:
                rows.extend(dict(run=run, **c) for c in cases(env, r[1], cfg.render, a.min_shift, a.min_mass,
                                                                    a.sanity))
            if (j + 1) % 25 == 0:
                print(f"  {run}: {j + 1}/{len(idx)} windows", end="\r")
        print()
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "action_gain.csv")
    df.to_csv(out, index=False)
    report(df)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
