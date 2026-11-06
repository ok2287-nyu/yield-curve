"""Mass, sharpness and error of several checkpoints on the SAME validation windows (one prior sample each).

    python scripts/eval_sharpness.py --cache ../round_cache \
        --ckpts runs/cnn_det/best.pt runs/ws_both_long/last.pt runs/ws_ol20_b/last.pt --n_windows 500

Per model, at 1 / 2 / 4 s:
    mass    predicted / true green          (falling: cars fading)
    sharp   sharpness / the truth's          sharpness = sum g^2 / sum g, the mass-weighted mean intensity
                                             (falling: cars smeared, dimmed, or hedged across places)
    mse     green-channel MSE of the sample  (where the cars are)
High mass with low sharpness = blur; low mass with high sharpness = cars vanish but the rest stay solid.
CPU is fine (~a minute per model per 500 windows for the CNN). Same windows and noise seeds for every model.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.wm_eval import sharpness_sums

REPORT = (5, 10, 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]

    rows = []
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=T)
        idx = np.sort(np.random.default_rng(a.seed).choice(len(ds), min(a.n_windows, len(ds)), replace=False))
        acc = {k: torch.zeros(T) for k in ("sq_p", "m_p", "sq_t", "m_t", "se")}
        with torch.no_grad():
            for j in range(0, len(idx), a.batch):
                items = [ds[int(i)] for i in idx[j:j + a.batch]]
                b = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}
                torch.manual_seed(a.seed + j)                              # same noise for every model
                g = m(b["images"], b["actions"], b["speed"], T, z_source="prior")["green"].float().cpu()
                y = b["images"][:, n:n + T, GREEN:GREEN + 1].float().cpu()
                q, s_ = sharpness_sums(g); qt, st = sharpness_sums(y)
                acc["sq_p"] += q; acc["m_p"] += s_; acc["sq_t"] += qt; acc["m_t"] += st
                acc["se"] += ((g - y) ** 2).mean((2, 3, 4)).sum(0)
        mass = acc["m_p"] / acc["m_t"].clamp_min(1e-8)
        sharp = (acc["sq_p"] / acc["m_p"].clamp_min(1e-8)) / (acc["sq_t"] / acc["m_t"].clamp_min(1e-8))
        mse = acc["se"] / len(idx)
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}/{os.path.basename(path)}"
        rows.append((name, mass, sharp, mse))
        print(f"done {name} ({'deterministic' if m.cfg.deterministic else 'stochastic'}, step {s.get('step', '?')})")

    print(f"\n{len(idx)} {a.split} windows, one prior sample each: mass / sharpness (vs truth) / MSE")
    print(f"{'':28s}" + "".join(f"{f'{k * dt:.0f} s':>26s}" for k in REPORT))
    for name, mass, sharp, mse in rows:
        print(f"{name:28s}" + "".join(f"{f'm {mass[k-1]:.2f}  sh {sharp[k-1]:.2f}  {mse[k-1]:.4f}':>26s}"
                                      for k in REPORT))


if __name__ == "__main__":
    main()
