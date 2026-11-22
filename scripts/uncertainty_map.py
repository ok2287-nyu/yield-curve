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
--occ_only: the world maps average only over cells that hold a road user in the true frame. Without it every map
mostly shows WHERE TRAFFIC DRIVES (empty cells are certain and dilute the rest); with it, where on the roundabout
the model is unsure ABOUT THE ROAD USERS. Also prints the --top world bins with the highest mean of each map.
Writes, next to the first checkpoint unless --out_dir:
    uncertainty_<run>.png   the maps averaged in WORLD coordinates (3 m bins) over the roundabout photo
    uncertainty_<run>.npz   the world grids (sum, count) for each map
    uncertainty_stats.csv   the numbers above
Confound checks (all off by default):
    --edge_cells N   drop cells within N cells (6 m each) of the crop border, everywhere (stats, maps, hotspots):
                     a road user entering the 48 m view is unknowable, which inflates the map at the arm ends.
    --motion_bins Q  "motion" = per-cell mean |true green(t+1) - true green(t) warped by the ego motion|: how much the
                     road users in the cell moved, in pixels (the warp-previous residual). Fast road users move more
                     pixels, so every map rises with motion whatever the model knows. Prints, within road-user cells,
                     the maps per motion quantile and the Spearman of sigma_p with err after removing motion (rank
                     residuals), and writes uncertainty_<run><tag>_motion.png: motion, sigma_p / its motion-bin mean,
                     err / its motion-bin mean. A bin that stays hot in the middle panel is uncertain for its motion.
    --fork_xy X Y    fork or blur? picks the --fork_n road-user cell-steps within --fork_r m of world (X, Y) with the
                     highest sigma_p (one per window), reruns --fork_k prior rollouts and writes fork_<run>.png
                     (truth | samples | sample mean, 32 px around the cell, at that step) and prints per sample the
                     green mass, peak and centroid in the patch. A fork: sharp samples (peak near the truth's) at
                     different places. A blur: low peaks, mass smeared.
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
def sample_maps(m, b, n, H, k, return_g=False):
    """k prior rollouts -> (spread, err, occ): (B, H, 8, 8) each (+ the rollouts (k, B, H, 64, 64) if return_g)."""
    from ladder import scheduled_rollout
    g = torch.stack([scheduled_rollout(m, b, n, H, lambda s: "prior") for _ in range(k)])      # (k, B, H, 64, 64)
    tgt = b["images"][:, n:n + H, GREEN].float()                                             # (B, H, 64, 64)
    pool = lambda x: F.avg_pool2d(x.flatten(0, 1)[:, None], 8)[:, 0].view(*x.shape[:2], 8, 8)
    spread = pool(g.std(0, unbiased=True))
    err = pool((g.mean(0) - tgt) ** 2)
    occ = F.max_pool2d(tgt.flatten(0, 1)[:, None], 8)[:, 0].view(*tgt.shape[:2], 8, 8) > 0.1
    if return_g:
        return spread, err, occ, g
    return spread, err, occ


@torch.no_grad()
def true_motion(b, n, H, rc):
    """(B, H, 8, 8): mean over each cell of |true green(n+k) - true green(n+k-1) warped into frame n+k by the logged
    ego poses|, i.e. how much the road users there moved in 0.2 s (the warp-previous residual)."""
    from roundabout_wm.wm_eval import warp_previous
    imgs, pose = b["images"], b["ego_state"][..., :3]
    out = []
    for k in range(H):
        i = n - 1 + k
        w = warp_previous(imgs[:, i, GREEN:GREEN + 1], pose[:, i], pose[:, i + 1:i + 2], rc)[:, 0, 0]
        out.append((imgs[:, i + 1, GREEN].float() - w).abs())
    d = torch.stack(out, 1)                                                                  # (B, H, 64, 64)
    return F.avg_pool2d(d.flatten(0, 1)[:, None], 8)[:, 0].view(*d.shape[:2], 8, 8)


def edge_keep(B, H, edge_cells):
    """(B*H*64,) bool: False for cells within edge_cells cells of the crop border."""
    r = np.arange(8)
    d = np.minimum.outer(np.minimum(r, 7 - r), np.minimum(r, 7 - r))                        # (8, 8) cells to border
    return np.tile((d >= edge_cells).ravel(), B * H)


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
    p.add_argument("--occ_only", action="store_true", help="world maps over road-user cells only (files get _occ)")
    p.add_argument("--top", type=int, default=8, help="print this many hotspot bins per map")
    p.add_argument("--edge_cells", type=int, default=0, help="drop cells this many cells from the crop border")
    p.add_argument("--motion_bins", type=int, default=0, help="motion-quantile control (0 = off)")
    p.add_argument("--fork_xy", type=float, nargs=2, default=None, metavar=("X", "Y"), help="fork/blur sheet here")
    p.add_argument("--fork_r", type=float, default=4.5, help="m around --fork_xy")
    p.add_argument("--fork_n", type=int, default=6, help="windows in the fork sheet")
    p.add_argument("--fork_k", type=int, default=8, help="prior rollouts per window in the fork sheet")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, H, rc = cfg.data.n_cond, a.horizon, cfg.render
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.ckpts[0]))
    os.makedirs(out_dir, exist_ok=True)
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
        cols = {k: [] for k in MAPS + ("occ", "motion", "win", "step", "cell", "wx", "wy", "gx", "gy", "inb")}
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
            vals["motion"] = true_motion(b, n, H, rc)
            vals = {k: v.cpu().numpy() for k, v in vals.items()}
            if sig is not None:
                sg = vals["sigma_p"].reshape(B * H, 64)
                within.append(sg.std(1) / np.maximum(sg.mean(1), 1e-8))
            keep = edge_keep(B, H, a.edge_cells)
            # world grid
            st = b["ego_state"][:, n:n + H, :3].cpu().numpy().reshape(B * H, 3)
            xy = cell_world_xy(st, rc).reshape(-1, 2)
            gx = np.floor(xy[:, 0] / a.bin).astype(int)
            gy = np.floor(-xy[:, 1] / a.bin).astype(int)
            ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            extra = {"occ": occ.cpu().numpy().ravel(), "win": np.repeat(idx[j:j + a.batch], H * 64),
                     "step": np.tile(np.repeat(np.arange(H), 64), B), "cell": np.tile(np.arange(64), B * H),
                     "wx": xy[:, 0], "wy": xy[:, 1], "gx": gx, "gy": gy, "inb": ok}
            for k in MAPS + ("motion",):
                cols[k].append(vals[k].ravel()[keep])
            for k, v in extra.items():
                cols[k].append(v[keep])
            ok = ok & keep
            if a.occ_only:
                ok &= occ.cpu().numpy().reshape(-1)
            np.add.at(cnt, (gy[ok], gx[ok]), 1)
            for k in MAPS:
                v = vals[k].reshape(-1)
                if np.isfinite(v).any():
                    np.add.at(grid[k], (gy[ok], gx[ok]), np.nan_to_num(v[ok]))
        df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
        print(f"\n{run} ({'fixed' if m.cfg.fixed_prior else 'learned'} prior, "
              f"{'1-D / sequence z' if (m.cfg.z_global or m.cfg.z_seq) else 'spatial z'}, step {s.get('step', '?')}): "
              f"{len(idx)} windows, {H} steps ({H * cfg.data.dt:.1f} s), k = {a.k}"
              + (f", cells within {a.edge_cells} of the border dropped" if a.edge_cells else ""))
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
        tag = ("_occ" if a.occ_only else "") + (f"_e{a.edge_cells}" if a.edge_cells else "")
        np.savez(os.path.join(out_dir, f"uncertainty_{run}{tag}.npz"), count=cnt, bin=a.bin, bg_m_per_px=bg_m,
                 **{k: grid[k] for k in MAPS})
        plot(run, grid, cnt, bg, bg_m, a, os.path.join(out_dir, f"uncertainty_{run}{tag}.png"))
        hotspots(grid, cnt, a)
        if a.motion_bins:
            row.update(motion_check(df, run, bg, bg_m, a, (ny, nx), os.path.join(out_dir, f"uncertainty_{run}{tag}_motion.png")))
        if a.fork_xy is not None:
            fork_sheet(m, ds, df, run, a, cfg, dev, os.path.join(out_dir, f"fork_{run}{tag}.png"))
    if stats:
        pd.DataFrame(stats).to_csv(os.path.join(out_dir, "uncertainty_stats.csv"), index=False)
        print(f"\nwrote uncertainty_<run>.png / .npz and uncertainty_stats.csv to {out_dir}")


def hotspots(grid, cnt, a, keys=MAPS):
    """Print the --top world bins (>= --min_count samples) with the highest mean of each map, with their world
    position (m; x right, y up as in the background photo) and the bin's rank among all bins."""
    seen = cnt >= a.min_count
    where = "road-user cells" if a.occ_only else "all cells"
    print(f"  hotspots ({where}, {a.bin:.0f} m bins with >= {a.min_count} samples, {seen.sum()} bins): "
          f"x, y = bin centre in m, mean, n")
    for k in keys:
        mean = np.where(seen, grid[k] / np.maximum(cnt, 1), np.nan)
        if not np.isfinite(mean).any():
            continue
        flat = np.argsort(np.nan_to_num(mean, nan=-np.inf).ravel())[::-1][:a.top]
        ys, xs = np.unravel_index(flat, mean.shape)
        med = np.nanmedian(mean)
        cells = ", ".join(f"({(x + .5) * a.bin:.0f}, {-(y + .5) * a.bin:.0f}) {mean[y, x]:.3g} n{int(cnt[y, x])}"
                          for y, x in zip(ys, xs))
        print(f"    {k:8s} median bin {med:.3g} | top: {cells}")


def _rank_resid(y, x):
    """Rank-transform y and x, remove the linear fit of rank(y) on rank(x): rank(y) with x taken out."""
    ry, rx = pd.Series(y).rank().to_numpy(), pd.Series(x).rank().to_numpy()
    rx = rx - rx.mean()
    return ry - ry.mean() - rx * (rx @ (ry - ry.mean())) / max(rx @ rx, 1e-12)


def motion_check(df, run, bg, bg_m, a, shape, path):
    """Within road-user cells: the maps per motion quantile, the sigma_p-err Spearman with motion taken out, and
    world maps of motion and of sigma_p / err relative to their motion-bin mean."""
    d = df[df.occ & np.isfinite(df.sigma_p)].copy()
    if len(d) < 10 * a.motion_bins:
        print("  (too few road-user cells for the motion check)"); return {}
    d["mbin"] = pd.qcut(d.motion.rank(method="first"), a.motion_bins, labels=False)
    print(f"  motion control (road-user cells, {a.motion_bins} motion quantiles):")
    print(f"    {'bin':>3s} {'n':>7s} {'motion':>7s} {'sigma_p':>8s} {'spread':>7s} {'err':>7s} {'rho(sig,err)':>13s}")
    for q, g in d.groupby("mbin"):
        rho = g[["sigma_p", "err"]].corr(method="spearman").iloc[0, 1]
        print(f"    {q:3d} {len(g):7d} {g.motion.mean():7.4f} {g.sigma_p.mean():8.3f} {g.spread.mean():7.4f} "
              f"{g.err.mean():7.4f} {rho:13.2f}")
    rho_m = {k: d[[k, "motion"]].corr(method="spearman").iloc[0, 1] for k in ("sigma_p", "spread", "err")}
    partial = float(np.corrcoef(_rank_resid(d.sigma_p, d.motion), _rank_resid(d.err, d.motion))[0, 1])
    raw = d[["sigma_p", "err"]].corr(method="spearman").iloc[0, 1]
    print("    Spearman with motion: " + ", ".join(f"{k} {v:.2f}" for k, v in rho_m.items()))
    print(f"    sigma_p vs err: {raw:.2f} raw, {partial:.2f} with motion taken out (rank residuals)")
    # world maps relative to the motion bin
    for k in ("sigma_p", "err"):
        d[k + "_rel"] = d[k] / d.groupby("mbin")[k].transform("mean")
    d = d[d.inb]
    ny, nx = shape
    cnt = np.zeros(shape)
    np.add.at(cnt, (d.gy.to_numpy(), d.gx.to_numpy()), 1)
    grids = {}
    for k in ("motion", "sigma_p_rel", "err_rel"):
        grids[k] = np.zeros(shape)
        np.add.at(grids[k], (d.gy.to_numpy(), d.gx.to_numpy()), d[k].to_numpy())
    titles = {"motion": "motion: |true change| not explained by ego motion",
              "sigma_p_rel": "prior sigma / mean for its motion bin", "err_rel": "error / mean for its motion bin"}
    _world_panels(grids, cnt, titles, bg, bg_m, a, f"{run}: road-user cells, motion-controlled", path,
                  center={"sigma_p_rel": 1.0, "err_rel": 1.0})
    sub = argparse.Namespace(**{**vars(a), "occ_only": True})
    print("  hotspots after the motion control (value 1 = as uncertain as its motion predicts):")
    hotspots(grids, cnt, sub, keys=("sigma_p_rel", "err_rel"))
    return {"rho_occ_sigma_err_motion_out": partial, **{f"rho_occ_{k}_motion": v for k, v in rho_m.items()}}


@torch.no_grad()
def fork_sheet(m, ds, df, run, a, cfg, dev, path):
    """Fork or blur at world (X, Y): the --fork_n highest-sigma_p road-user cell-steps nearby, one per window."""
    from ladder import scheduled_rollout
    X, Y = a.fork_xy
    d = df[df.occ & np.isfinite(df.sigma_p) & (np.hypot(df.wx - X, df.wy - Y) <= a.fork_r)]
    if d.empty:
        print(f"  fork: no road-user cell within {a.fork_r} m of ({X}, {Y})"); return
    d = d.sort_values("sigma_p", ascending=False).drop_duplicates("win").head(a.fork_n)
    n, P = cfg.data.n_cond, 16
    rows = []
    print(f"  fork / blur at ({X:.0f}, {Y:.0f}), {len(d)} windows, k = {a.fork_k}: per sample green mass, peak, "
          f"centroid (m from the cell centre, forward / left)")
    for _, r in d.iterrows():
        it = ds[int(r.win)]
        b = {key: v[None].to(dev) for key, v in it.items()}
        step = int(r.step)
        torch.manual_seed(a.seed + int(r.win))
        g = torch.stack([scheduled_rollout(m, b, n, step + 1, lambda s: "prior")[0, step] for _ in range(a.fork_k)])
        tru = b["images"][0, n + step, GREEN].float()
        cr, cc = divmod(int(r.cell), 8)
        r0, c0 = cr * 8 + 4 - P, cc * 8 + 4 - P
        pad = lambda x: F.pad(x, (P, P, P, P))[..., r0 + P:r0 + 3 * P, c0 + P:c0 + 3 * P]
        pt, ps = pad(tru).cpu().numpy(), pad(g).cpu().numpy()                     # (2P, 2P), (k, 2P, 2P)
        rr, cc_ = np.mgrid[0:2 * P, 0:2 * P]

        def desc(x):
            mass = x.sum()
            if mass < 1e-3:
                return f"{mass:5.1f} {x.max():.2f}      -"
            fr = (P - 0.5 - (x * rr).sum() / mass) * cfg.render.m_per_px
            lf = (P - 0.5 - (x * cc_).sum() / mass) * cfg.render.m_per_px
            return f"{mass:5.1f} {x.max():.2f} {fr:+5.1f}/{lf:+5.1f}"
        print(f"    window {int(r.win)} step {step + 1} ({(step + 1) * 0.2:.1f} s) sigma_p {r.sigma_p:.2f} "
              f"world ({r.wx:.0f}, {r.wy:.0f})")
        print(f"      truth   {desc(pt)}")
        for i in range(len(ps)):
            print(f"      sample{i} {desc(ps[i])}")
        cen = []
        for x in ps:
            if x.sum() > 1e-3:
                cen.append([(x * rr).sum() / x.sum(), (x * cc_).sum() / x.sum()])
        cen = np.array(cen) * cfg.render.m_per_px
        sp = float(np.sqrt(((cen - cen.mean(0)) ** 2).sum(1).mean())) if len(cen) > 1 else float("nan")
        print(f"      peak: truth {pt.max():.2f}, samples mean {ps.max((1, 2)).mean():.2f}; sample mean peak "
              f"{ps.mean(0).max():.2f}; centroid scatter {sp:.1f} m")
        rows.append((pt, ps, f"win {int(r.win)} t+{(step + 1) * 0.2:.1f}s sig {r.sigma_p:.2f}"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = a.fork_k
    fig, axes = plt.subplots(len(rows), K + 2, figsize=(1.6 * (K + 2), 1.7 * len(rows)), squeeze=False)
    for i, (pt, ps, lab) in enumerate(rows):
        for j, (img, t) in enumerate([(pt, "truth")] + [(x, f"s{q}") for q, x in enumerate(ps)] +
                                     [(ps.mean(0), "mean")]):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.add_patch(plt.Rectangle((P - 4.5, P - 4.5), 8, 8, fill=False, ec="r", lw=0.6))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(t, fontsize=8)
        axes[i, 0].set_ylabel(lab, fontsize=7)
    fig.suptitle(f"{run}: fork or blur at ({X:.0f}, {Y:.0f}); green channel, 24 m around the cell (red box); "
                 f"ego forward is up", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"  wrote {path}")


def _world_panels(grids, cnt, titles, bg, bg_m, a, suptitle, path, center=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext_bg = [0, bg.shape[1] * bg_m, -bg.shape[0] * bg_m, 0]
    ny, nx = cnt.shape
    ext = [0, nx * a.bin, -ny * a.bin, 0]
    seen = cnt >= a.min_count
    if not seen.any():
        print(f"  (no world bin has >= {a.min_count} samples; map not drawn)"); return
    ys, xs = np.where(seen)
    zoom = [xs.min() * a.bin - 10, (xs.max() + 1) * a.bin + 10, -(ys.max() + 1) * a.bin - 10, -ys.min() * a.bin + 10]
    fig, axes = plt.subplots(1, len(grids), figsize=(5.5 * len(grids), 6), squeeze=False)
    for ax, (k, gsum) in zip(axes[0], grids.items()):
        ax.imshow(bg, cmap="gray", extent=ext_bg, alpha=0.6)
        mean = np.where(seen, gsum / np.maximum(cnt, 1), np.nan)
        lo, hi = np.nanpercentile(mean, [5, 95])
        if center and k in center:
            w = max(abs(hi - center[k]), abs(center[k] - lo), 1e-6)
            im = ax.imshow(mean, extent=ext, alpha=0.75, cmap="RdBu_r", vmin=center[k] - w, vmax=center[k] + w,
                           interpolation="nearest")
        else:
            im = ax.imshow(mean, extent=ext, alpha=0.65, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
        ax.set_title(titles[k]); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


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
    fig.suptitle(f"{run}: mean over {a.horizon} steps ({a.horizon * 0.2:.1f} s), world bins of {a.bin:.0f} m"
                 + (", road-user cells only" if a.occ_only else ""))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
