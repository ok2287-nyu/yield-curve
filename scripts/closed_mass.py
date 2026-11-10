"""Do prior rollouts really lose cars, or draw them in the wrong place? Total green in CLOSED windows.

    python scripts/closed_mass.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/cnn_det/best.pt runs/ab_anchor/last.pt runs/ab_fixedprior/last.pt --k 4

ladder.py's "kept" counts green only within +-1 s of each car's true path, so a sharp car drawn 1.5 s late scores as
faded + invented. This script ignores position. It keeps only CLOSED windows: the same road users are in view, whole
(every box corner at least --margin m inside the crop), from the last warm-up frame to the end of the rollout, and no
other road user touches the crop. Then the true green mass is constant-ish and fully explained by seen cars, and
    mass  = sum of predicted green / sum of true green        (per step, pooled over windows)
is a position-free retention measure: ~1 = cars kept (wherever drawn), < 1 = cars really vanish, > 1 = invented.
Also per rung:
    sharp   sum g^2 / sum g relative to the truth's (1 = as solid as the truth)
    spread  mean |g_a - g_b| between two prior samples of the same window / mean true green (0 = samples identical:
            no diversity). Only for roll_prior with --k >= 2.
Rungs: roll_post (posterior z), roll_prior (k prior samples), roll_pmean (prior mean; z = 0 for --fixed_prior).
Deterministic models: one row (z = 0). CPU is fine. Rendered rounD data: run on HPC / your own machine only.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.render import car_corners

REPORT = (5, 10, 15, 20)


def inside_frac(corners, ex, ey, eh, ahead, behind, half, margin):
    """corners (N, 4, 2) -> (any corner touches the crop (N,), all corners >= margin inside (N,))."""
    d = corners - np.array([ex, ey])
    c, s = np.cos(eh), np.sin(eh)
    fwd = d[..., 0] * c + d[..., 1] * s
    left = -d[..., 0] * s + d[..., 1] * c
    touch = (fwd <= ahead) & (fwd >= -behind) & (np.abs(left) <= half)
    whole = (fwd <= ahead - margin) & (fwd >= -behind + margin) & (np.abs(left) <= half - margin)
    return touch.any(1), whole.all(1)


def is_closed(win, st, n, box, margin, min_users=1):
    """win: list of n+T DataFrames (road users per frame, ego dropped); st (n+T, >=3) ego x, y, heading.
    Closed: the road users whole in view at frame n-1 stay whole in view through the end, and nobody else touches
    the crop at frames n-1 .. end. -> (closed, number of road users)."""
    S = None
    for j in range(n - 1, len(win)):
        g = win[j]
        if len(g):
            cor = car_corners(g["x"].to_numpy(), g["y"].to_numpy(), g["heading"].to_numpy(),
                              g["length"].to_numpy(), g["width"].to_numpy())
            touch, whole = inside_frac(cor, st[j, 0], st[j, 1], st[j, 2], *box, margin)
            touching, wholes = set(g.index[touch].tolist()), set(g.index[whole].tolist())
        else:
            touching, wholes = set(), set()
        if S is None:
            S = wholes
            if len(S) < min_users or touching != S:
                return False, len(S)
        elif touching != S or wholes != S:
            return False, len(S)
    return True, len(S)


@torch.no_grad()
def rollouts(m, b, n, T, k):
    """-> dict rung -> list of (B, T, 64, 64)."""
    from ladder import scheduled_rollout
    if m.cfg.deterministic:
        return {"det": [scheduled_rollout(m, b, n, T, lambda s: "prior")]}
    return {"roll_post": [scheduled_rollout(m, b, n, T, lambda s: "post")],
            "roll_prior": [scheduled_rollout(m, b, n, T, lambda s: "prior") for _ in range(k)],
            "roll_pmean": [scheduled_rollout(m, b, n, T, lambda s: "pmean")]}


def main():
    from fov_popin import crops
    from leakage import WindowTracks
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_scan", type=int, default=0, help="windows to scan for closed ones (0 = the whole split)")
    p.add_argument("--n_windows", type=int, default=500, help="at most this many closed windows are scored")
    p.add_argument("--margin", type=float, default=1.0, help="m a whole box must stay inside the crop edge")
    p.add_argument("--k", type=int, default=4, help="prior samples per window")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    s0 = torch.load(a.ckpts[0], map_location="cpu")
    ds0 = RounDWindows(a.cache, split, cfg, s0["stats"], n_pred=T)
    order = np.random.default_rng(a.seed).permutation(len(ds0))
    if a.n_scan:
        order = order[:a.n_scan]
    tracks = WindowTracks(SimpleNamespace(cfg=cfg, ds=ds0, rec_ids=split), a.cache, a.data_dir)
    box = crops(cfg)["current"]
    idx, users = [], []
    for c, i in enumerate(order):
        win, st = tracks.window(int(i))
        ok, nu = is_closed(win, st, n, box, a.margin)
        if ok:
            idx.append(int(i)); users.append(nu)
            if len(idx) >= a.n_windows:
                break
        if (c + 1) % 500 == 0:
            print(f"  scanned {c + 1} windows, {len(idx)} closed")
    idx = np.sort(np.array(idx, dtype=int))
    print(f"{len(idx)} closed {a.split} windows (scanned {c + 1}); road users per window: "
          f"mean {np.mean(users) if users else 0:.1f}, max {max(users) if users else 0}; margin {a.margin} m")
    if len(idx) == 0:
        return

    rows = []
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=T)
        acc = {}
        for j in range(0, len(idx), a.batch):
            items = [ds[int(i)] for i in idx[j:j + a.batch]]
            b = {key: torch.stack([it[key] for it in items]).to(dev) for key in items[0]}
            tgt = b["images"][:, n:n + T, GREEN].float()                                 # (B, T, 64, 64)
            torch.manual_seed(a.seed + j)
            for r, preds in rollouts(m, b, n, T, a.k).items():
                d = acc.setdefault(r, {key: np.zeros(T) for key in ("pred", "true", "sq", "sq_t", "l1", "n_pair")})
                for g in preds:
                    d["pred"] += g.sum((0, 2, 3)).cpu().numpy(); d["true"] += tgt.sum((0, 2, 3)).cpu().numpy()
                    d["sq"] += (g * g).sum((0, 2, 3)).cpu().numpy(); d["sq_t"] += (tgt * tgt).sum((0, 2, 3)).cpu().numpy()
                for x in range(len(preds)):
                    for y in range(x + 1, len(preds)):
                        d["l1"] += (preds[x] - preds[y]).abs().sum((0, 2, 3)).cpu().numpy()
                        d["n_pair"] += 1
                d.setdefault("true1", np.zeros(T))
                d["true1"] += tgt.sum((0, 2, 3)).cpu().numpy()
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}/{os.path.basename(path)}"
        print(f"\n{name} ({'deterministic' if m.cfg.deterministic else 'stochastic'}, step {s.get('step', '?')})")
        print(f"{'rung':11s} {'time':>5s} {'mass':>6s} {'sharp':>6s} {'spread':>7s}")
        for r, d in acc.items():
            for t in range(T):
                mass = d["pred"][t] / max(d["true"][t], 1e-8)
                sharp = (d["sq"][t] / max(d["pred"][t], 1e-8)) / (d["sq_t"][t] / max(d["true"][t], 1e-8))
                spread = (d["l1"][t] / d["n_pair"][t]) / max(d["true1"][t], 1e-8) if d["n_pair"][t] else float("nan")
                rows.append(dict(ckpt=name, rung=r, time_s=round((t + 1) * dt, 2), mass=mass, sharp=sharp,
                                 spread=spread, n_windows=len(idx)))
                if t + 1 in REPORT:
                    sp = f"{spread:7.3f}" if spread == spread else f"{'-':>7s}"
                    print(f"{r:11s} {(t + 1) * dt:4.1f}s {mass:6.2f} {sharp:6.2f} {sp}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "closed_mass.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nspread: mean |sample_a - sample_b| per pixel / mean true green (0 = identical samples)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
