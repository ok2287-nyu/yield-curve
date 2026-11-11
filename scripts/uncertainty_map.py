"""Where is the model unsure? Per-cell uncertainty maps, how they relate, and where they sit on the roundabout.

    python scripts/uncertainty_map.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_anchor/last.pt runs/ab_fixedprior/last.pt --n_windows 500 --k 8

For every 8x8 cell (6 m x 6 m) and rollout step up to --horizon (default 10 steps = 2 s), four numbers:
    sigma_p  the prior's width in that cell, mean over the cell's z channels of exp(logvar_p / 2). Learned-prior
             models only (constant 1 with --fixed_prior). Taken on the posterior-driven track, i.e. with h kept on
             the real scene, so the cell matches the real road users.
    kl       KL(q || p) summed over the cell's z channels, same track: how much the real next frame surprised the
             prior in that cell. Needs the real future, but not a calibrated sigma_p.
    spread   std over --k prior rollouts of the predicted green, averaged over the cell's 64 pixels: how much the
             model's own imagined futures disagree there (no future frame needed: what a controller would see).
    err      squared error of the MEAN of the k prior rollouts against the truth, averaged over the cell.
Plus occ: the cell holds a road user in the true target frame.

Prints, per checkpoint:
    - sigma_p's spread: pooled percentiles, and the within-scene variation (std over the 64 cells of one window and
      step / mean, averaged). ~0 = a flat map that says nothing.
    - means of each map in cells with and without a road user
    - Spearman correlations between the maps (over all cell-steps). A meaningful uncertainty map should rise where
      samples spread and where the model is wrong.
    - the same correlations within road-user cells only: every map is larger where a road user is, so this is the
      test of whether a map says WHICH road users are uncertain, not just where road users are.
Writes, next to the first checkpoint unless --out_dir:
    uncertainty_<run>.png   the maps averaged in WORLD coordinates (3 m bins) over the roundabout photo
    uncertainty_<run>.npz   the world grids (sum, count) for each map
    uncertainty_stats.csv   the numbers above
The PNGs contain the rounD background photo: keep them on HPC / your own machine. CPU is fine.
Spatial-z models only for sigma_p and kl (1-D / per-sequence z: spread and err only); deterministic models skipped.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel, gauss_kl  # noqa: F401

MAPS = ("sigma_p", "kl", "spread", "err")


def cell_world_xy(ego_xyh, rc):
    """ego_xyh (N, 3) world x, y, heading -> (N, 8, 8, 2) world x, y of each 8x8 cell's centre."""
    r = np.arange(8) * 8 + 3.5                                            # cell-centre pixel row / column
    rows, cols = np.meshgrid(r, r, indexing="ij")
    fwd = (rc.ego_row - rows) * rc.m_per_px                               # (8, 8)
    left = (rc.ego_col - cols) * rc.m_per_px
    x, y, h = (ego_xyh[:, i, None, None] for i in range(3))
    c, s = np.cos(h), np.sin(h)
    return np.stack([x + fwd * c - left * s, y + fwd * s + left * c], -1)


@torch.no_grad()
def posterior_track(m, b, n, H):
    """h driven by posterior samples. -> (sigma_p, kl): (B, H, 8, 8) each, or (None, None) without a spatial z."""
    if m.cfg.z_global or m.cfg.z_seq:
        return None, None
    imgs, act, spd = b["images"], b["actions"], b["speed"]
    e = m.encode(imgs[:, :n + H])
    h, _ = m.warmup(e[:, :n], act, spd)
    prev = imgs[:, n - 1, GREEN:GREEN + 1]
    sig, kls = [], []
    for k in range(H):
        i = n - 1 + k
        frame = None
        if m.cfg.reencode:
            frame = imgs[:, i] if k == 0 else torch.cat([imgs[:, i, :GREEN], prev.to(imgs.dtype),
                                                         imgs[:, i, GREEN + 1:]], 1)
        mu_p, lv_p = m.prior_params(h)
        mu_q, lv_q = m.posterior(torch.cat([h, e[:, i + 1]], 1))
        mu_q, lv_q, mu_p, lv_p = (t.float() for t in (mu_q, lv_q, mu_p, lv_p))
        sig.append((0.5 * lv_p).exp().mean(1))
        kls.append(0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).sum(1))
        z = m._sample(mu_q, lv_q)
        h, g = m.rollout_step(h, z, act[:, i], spd[:, i], frame, prev)
        prev = g
    return torch.stack(sig, 1), torch.stack(kls, 1)


@torch.no_grad()
def sample_maps(m, b, n, H, k):
    """k prior rollouts -> (spread, err, occ): (B, H, 8, 8) each."""
    from ladder import scheduled_rollout
    g = torch.stack([scheduled_rollout(m, b, n, H, lambda s: "prior") for _ in range(k)])      # (k, B, H, 64, 64)
    tgt = b["images"][:, n:n + H, GREEN].float()                                             # (B, H, 64, 64)
    pool = lambda x: F.avg_pool2d(x.flatten(0, 1)[:, None], 8)[:, 0].view(*x.shape[:2], 8, 8)
    spread = pool(g.std(0, unbiased=True))
    err = pool((g.mean(0) - tgt) ** 2)
    occ = F.max_pool2d(tgt.flatten(0, 1)[:, None], 8)[:, 0].view(*tgt.shape[:2], 8, 8) > 0.1
    return spread, err, occ


def background(data_dir, rec, cfg):
    import cv2
    meta = pd.read_csv(os.path.join(data_dir, f"{rec:02d}_recordingMeta.csv"))
    m = float(meta["orthoPxToMeter"].iloc[0]) * cfg.data.background_scale_down
    img = cv2.imread(os.path.join(data_dir, f"{rec:02d}_background.png"), cv2.IMREAD_GRAYSCALE)
    return img, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=500)
    p.add_argument("--k", type=int, default=8, help="prior rollouts per window for spread / err")
    p.add_argument("--horizon", type=int, default=10, help="rollout steps used (10 = 2 s)")
    p.add_argument("--bin", type=float, default=3.0, help="world grid bin, m")
    p.add_argument("--min_count", type=int, default=20, help="world bins with fewer samples are not drawn")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_corr", type=int, default=200000, help="cell-steps used for the correlations")
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, H, rc = cfg.data.n_cond, a.horizon, cfg.render
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.ckpts[0]))
    bg, bg_m = background(a.data_dir, split[0], cfg)
    nx, ny = int(bg.shape[1] * bg_m / a.bin) + 1, int(bg.shape[0] * bg_m / a.bin) + 1    # world x in [0, W], y in [-H, 0]
    stats = []
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if m.cfg.deterministic:
            print(f"\n{run}: deterministic, no uncertainty; skipped"); continue
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=cfg.data.n_pred)
        idx = np.sort(np.random.default_rng(a.seed).choice(len(ds), min(a.n_windows, len(ds)), replace=False))
        grid = {k: np.zeros((ny, nx)) for k in MAPS}
        cnt = np.zeros((ny, nx))
        cols = {k: [] for k in MAPS + ("occ",)}
        within = []
        for j in range(0, len(idx), a.batch):
            items = [ds[int(i)] for i in idx[j:j + a.batch]]
            b = {key: torch.stack([it[key] for it in items]).to(dev) for key in items[0]}
            torch.manual_seed(a.seed + j)
            sig, kl = posterior_track(m, b, n, H)
            spread, err, occ = sample_maps(m, b, n, H, a.k)
            B = spread.shape[0]
            vals = {"spread": spread, "err": err,
                    "sigma_p": sig if sig is not None else torch.full_like(spread, float("nan")),
                    "kl": kl if kl is not None else torch.full_like(spread, float("nan"))}
            vals = {k: v.cpu().numpy() for k, v in vals.items()}
            if sig is not None:
                sg = vals["sigma_p"].reshape(B * H, 64)
                within.append(sg.std(1) / np.maximum(sg.mean(1), 1e-8))
            for k in MAPS:
                cols[k].append(vals[k].ravel())
            cols["occ"].append(occ.cpu().numpy().ravel())
            # world grid
            st = b["ego_state"][:, n:n + H, :3].cpu().numpy().reshape(B * H, 3)
            xy = cell_world_xy(st, rc).reshape(-1, 2)
            gx = np.floor(xy[:, 0] / a.bin).astype(int)
            gy = np.floor(-xy[:, 1] / a.bin).astype(int)
            ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            np.add.at(cnt, (gy[ok], gx[ok]), 1)
            for k in MAPS:
                v = vals[k].reshape(-1)
                if np.isfinite(v).any():
                    np.add.at(grid[k], (gy[ok], gx[ok]), np.nan_to_num(v[ok]))
        df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
        print(f"\n{run} ({'fixed' if m.cfg.fixed_prior else 'learned'} prior, "
              f"{'1-D / sequence z' if (m.cfg.z_global or m.cfg.z_seq) else 'spatial z'}, step {s.get('step', '?')}): "
              f"{len(idx)} windows, {H} steps ({H * cfg.data.dt:.1f} s), k = {a.k}")
        row = dict(run=run, fixed_prior=bool(m.cfg.fixed_prior))
        if within:
            q = np.nanpercentile(df["sigma_p"], [5, 50, 95])
            w = float(np.mean(np.concatenate(within)))
            print(f"  sigma_p: p5 {q[0]:.3f}  median {q[1]:.3f}  p95 {q[2]:.3f}  | within-scene variation "
                  f"(std over cells / mean) {w:.3f}")
            row.update(sigma_p5=q[0], sigma_p50=q[1], sigma_p95=q[2], sigma_within_cv=w)
        print(f"  {'map':8s} {'road-user cells':>16s} {'empty cells':>12s} {'ratio':>7s}")
        for k in MAPS:
            a1, a0 = df.loc[df.occ, k].mean(), df.loc[~df.occ, k].mean()
            if np.isfinite(a1):
                print(f"  {k:8s} {a1:16.4f} {a0:12.4f} {a1 / max(a0, 1e-12):7.2f}")
                row.update({f"{k}_occ": a1, f"{k}_empty": a0})
        sub = df.sample(min(len(df), a.max_corr), random_state=a.seed)
        use = [k for k in MAPS if np.isfinite(sub[k]).all() and sub[k].std() > 1e-9]
        corr = sub[use].corr(method="spearman")
        print("  Spearman correlations over cell-steps:")
        print("  " + corr.round(2).to_string().replace("\n", "\n  "))
        for x in use:
            for y in use:
                if x < y:
                    row[f"rho_{x}_{y}"] = corr.loc[x, y]
        # all maps are much larger where a road user is, so the correlations above are partly "is there a road
        # user here". Within road-user cells only: does the map still say WHICH road users are uncertain?
        occ_df = df[df.occ]
        sub = occ_df.sample(min(len(occ_df), a.max_corr), random_state=a.seed)
        use_o = [k for k in use if sub[k].std() > 1e-9]
        if len(sub) > 100 and len(use_o) > 1:
            corr_o = sub[use_o].corr(method="spearman")
            print(f"  Spearman correlations within road-user cells only ({len(occ_df)} cell-steps):")
            print("  " + corr_o.round(2).to_string().replace("\n", "\n  "))
            for x in use_o:
                for y in use_o:
                    if x < y:
                        row[f"rho_occ_{x}_{y}"] = corr_o.loc[x, y]
        stats.append(row)
        np.savez(os.path.join(out_dir, f"uncertainty_{run}.npz"), count=cnt, bin=a.bin, bg_m_per_px=bg_m,
                 **{k: grid[k] for k in MAPS})
        plot(run, grid, cnt, bg, bg_m, a, os.path.join(out_dir, f"uncertainty_{run}.png"))
    if stats:
        pd.DataFrame(stats).to_csv(os.path.join(out_dir, "uncertainty_stats.csv"), index=False)
        print(f"\nwrote uncertainty_<run>.png / .npz and uncertainty_stats.csv to {out_dir}")


def plot(run, grid, cnt, bg, bg_m, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext_bg = [0, bg.shape[1] * bg_m, -bg.shape[0] * bg_m, 0]                     # world x0, x1, y0, y1
    ny, nx = cnt.shape
    ext = [0, nx * a.bin, -ny * a.bin, 0]
    seen = cnt >= a.min_count
    if not seen.any():
        print(f"  (no world bin has >= {a.min_count} samples; map not drawn)"); return
    ys, xs = np.where(seen)
    zoom = [xs.min() * a.bin - 10, (xs.max() + 1) * a.bin + 10, -(ys.max() + 1) * a.bin - 10, -ys.min() * a.bin + 10]
    titles = {"sigma_p": "prior sigma (learned prior)", "kl": "KL: surprise by the real future",
              "spread": "spread between prior samples", "err": "error of the sample mean"}
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, k in zip(axes, MAPS):
        ax.imshow(bg, cmap="gray", extent=ext_bg, alpha=0.6)
        mean = np.where(seen, grid[k] / np.maximum(cnt, 1), np.nan)
        if np.isfinite(mean).any() and np.nanmax(mean) > np.nanmin(mean):
            lo, hi = np.nanpercentile(mean, [5, 95])
            im = ax.imshow(mean, extent=ext, alpha=0.65, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "flat / not available", transform=ax.transAxes, ha="center")
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
        ax.set_title(titles[k]); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{run}: mean over {a.horizon} steps ({a.horizon * 0.2:.1f} s), world bins of {a.bin:.0f} m")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
