"""How do the prior and the posterior compare on the SAME hidden state, step by step?

    python scripts/latent_eval.py --cache ../round_cache \
        --ckpts runs/ab_ctrl/last.pt runs/ab_anchor/last.pt runs/ab_anchor_z1d/last.pt --n_windows 500

At every rollout step k (state h_k, real next frame k+1), both heads are evaluated on the same h:
    prior      p(z | h)                  (N(0, I) for --fixed_prior models)
    posterior  q(z | h, Enc(frame k+1))
Per latent number j (spatial z: z_ch x 8 x 8 numbers; 1-D z: z_global numbers):
    gap   |mu_q - mu_p| / sigma_p      how far the posterior's centre sits from the prior's, in prior widths
    ratio sigma_q / sigma_p            < 1: the posterior is narrower (it knows something); ~1: as wide
    KL    KL(q || p) of that number    the information the posterior adds (nats)
    BC    Bhattacharyya coefficient    overlap of the two bells, 1 = identical, 0 = disjoint
Two tracks, so the drift question can be answered directly:
    post   h driven by posterior samples (as on most training steps): the prior on familiar states
    prior  h driven by prior samples (as in imagination): does the prior fall further behind as h drifts?
           (the posterior still sees the real next frame, so the KL here = what a correction would cost)
Spatial z only: numbers in cells where the NEXT frame has a road user ("car cells", green > 0.1 anywhere in the
6 m cell) vs empty cells, to see whether the information sits on the cars.
Prints, per checkpoint, track and step 1 / 5 / 10 / 20; writes latent_eval.csv next to the first checkpoint.
CPU is fine.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel

REPORT = (1, 5, 10, 20)
QS = (0.5, 0.9, 0.99)


def pair_stats(mu_q, lv_q, mu_p, lv_p):
    """Per-number comparison of two diagonal Gaussians (any equal shapes) -> dict of tensors of that shape."""
    mu_q, lv_q, mu_p, lv_p = (t.float() for t in (mu_q, lv_q, mu_p, lv_p))
    vq, vp = lv_q.exp(), lv_p.exp()
    dm2 = (mu_q - mu_p) ** 2
    return {"gap": dm2.sqrt() / vp.sqrt(),
            "log_ratio": 0.5 * (lv_q - lv_p),
            "kl": 0.5 * (lv_p - lv_q + (vq + dm2) / vp - 1),
            "bc": torch.sqrt(2 * (vq * vp).sqrt() / (vq + vp)) * torch.exp(-dm2 / (4 * (vq + vp)))}


def car_cells(green_next, z_shape):
    """green_next (B, 1, 64, 64) -> bool mask of z's shape (B, z_ch, 8, 8): the 8x8 cell holds a road user."""
    occ = F.max_pool2d(green_next.float(), 8) > 0.1                  # (B, 1, 8, 8)
    return occ.expand(z_shape)


@torch.no_grad()
def track(m, b, n, T, drive):
    """Roll out with z from `drive` ("post" or "prior"); at each step compare both heads on the same h.
    -> list over steps of (stats dict, car mask or None)."""
    imgs, act, spd = b["images"], b["actions"], b["speed"]
    e = m.encode(imgs[:, :n + T])
    h, _ = m.warmup(e[:, :n], act, spd)
    prev = imgs[:, n - 1, GREEN:GREEN + 1]
    out = []
    for k in range(T):
        i = n - 1 + k
        frame = None
        if m.cfg.reencode:
            frame = imgs[:, i] if k == 0 else torch.cat([imgs[:, i, :GREEN], prev.to(imgs.dtype),
                                                         imgs[:, i, GREEN + 1:]], 1)
        mu_p, lv_p = m.prior_params(h)
        mu_q, lv_q = m.posterior(torch.cat([h, e[:, i + 1]], 1))
        cars = None if m.cfg.z_global else car_cells(imgs[:, i + 1, GREEN:GREEN + 1], mu_q.shape)
        out.append((pair_stats(mu_q, lv_q, mu_p, lv_p), cars))
        z = m._sample(mu_q, lv_q) if drive == "post" else m._sample(mu_p, lv_p)
        h, g = m.rollout_step(h, z, act[:, i], spd[:, i], frame, prev)
        prev = g
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
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
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}/{os.path.basename(path)}"
        if m.cfg.deterministic:
            print(f"\n{name}: deterministic, no latent; skipped"); continue
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=T)
        idx = np.sort(np.random.default_rng(a.seed).choice(len(ds), min(a.n_windows, len(ds)), replace=False))
        # per track and step: sums, plus a random subsample of gaps for quantiles
        acc = {d: [dict(n_win=0, n_num=0, kl=0.0, gap=0.0, log_ratio=0.0, bc=0.0, kl_car=0.0, n_car=0,
                        bc_car=0.0, gap_car=0.0, gaps=[]) for _ in range(T)] for d in ("post", "prior")}
        g = torch.Generator().manual_seed(a.seed)
        for j in range(0, len(idx), a.batch):
            items = [ds[int(i)] for i in idx[j:j + a.batch]]
            b = {key: torch.stack([it[key] for it in items]).to(dev) for key in items[0]}
            for drive in ("post", "prior"):
                torch.manual_seed(a.seed + j)
                for k, (st, cars) in enumerate(track(m, b, n, T, drive)):
                    c = acc[drive][k]
                    B = st["kl"].shape[0]
                    c["n_win"] += B; c["n_num"] += st["kl"].numel()
                    c["kl"] += st["kl"].sum().item()
                    for key in ("gap", "log_ratio", "bc"):
                        c[key] += st[key].sum().item()
                    if cars is not None:
                        c["kl_car"] += st["kl"][cars].sum().item(); c["n_car"] += int(cars.sum())
                        c["bc_car"] += st["bc"][cars].sum().item(); c["gap_car"] += st["gap"][cars].sum().item()
                    flat = st["gap"].flatten().cpu()
                    pick = torch.randint(len(flat), (min(len(flat), 4000),), generator=g)
                    c["gaps"].append(flat[pick])
        kind = "1-D z" if m.cfg.z_global else "spatial z"
        prior = "fixed N(0, I) prior" if getattr(m.cfg, "fixed_prior", False) else "learned prior"
        print(f"\n{name} ({kind}, {prior}, step {s.get('step', '?')})")
        print(f"{'track':6s} {'time':>5s} {'KL/step':>8s} {'gap':>6s} {'gap50':>6s} {'gap90':>6s} {'gap99':>6s} "
              f"{'sq/sp':>6s} {'overlap':>8s} | {'KL/num car':>10s} {'KL/num empty':>12s} {'car share':>9s} "
              f"{'gap car':>8s} {'overlap car':>11s}")
        for drive in ("post", "prior"):
            for k in range(T):
                c = acc[drive][k]
                q = torch.quantile(torch.cat(c["gaps"]), torch.tensor(QS)).tolist()
                r = dict(ckpt=name, track=drive, time_s=round((k + 1) * dt, 2),
                         kl_step=c["kl"] / c["n_win"], gap=c["gap"] / c["n_num"],
                         **{f"gap_q{int(x * 100)}": v for x, v in zip(QS, q)},
                         ratio=float(np.exp(c["log_ratio"] / c["n_num"])), overlap=c["bc"] / c["n_num"])
                if c["n_car"]:
                    n_empty = c["n_num"] - c["n_car"]
                    r.update(kl_num_car=c["kl_car"] / c["n_car"],
                             kl_num_empty=(c["kl"] - c["kl_car"]) / max(n_empty, 1),
                             car_share_kl=c["kl_car"] / max(c["kl"], 1e-12), car_share_num=c["n_car"] / c["n_num"],
                             gap_car=c["gap_car"] / c["n_car"], overlap_car=c["bc_car"] / c["n_car"])
                rows.append(r)
                if k + 1 in REPORT:
                    car = (f"{r['kl_num_car']:10.3f} {r['kl_num_empty']:12.3f} "
                           f"{r['car_share_kl']:4.2f}/{r['car_share_num']:4.2f} {r['gap_car']:8.3f} "
                           f"{r['overlap_car']:11.3f}") if "kl_num_car" in r else f"{'-':>10s}"
                    print(f"{drive:6s} {(k + 1) * dt:4.1f}s {r['kl_step']:8.2f} {r['gap']:6.3f} {q[0]:6.3f} "
                          f"{q[1]:6.3f} {q[2]:6.3f} {r['ratio']:6.3f} {r['overlap']:8.3f} | {car}")
    if rows:
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "latent_eval.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\ncar share = share of the KL in car cells / share of numbers in car cells (spatial z only)")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
