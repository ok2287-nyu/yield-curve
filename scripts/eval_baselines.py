"""All baselines (and optionally a trained model) on the SAME validation windows, at 1 / 2 / 4 s.

    python scripts/eval_baselines.py --cache ../round_cache --data_dir ../round_data
    python scripts/eval_baselines.py --cache ../round_cache --data_dir ../round_data --ckpt runs/cnn_det/best.pt
    python scripts/eval_baselines.py ... --n_windows 0          # every val window (slower)

Rows (green-channel MSE, the same metric as training; plus the predicted / true green mass):
    empty   predict no road users
    copy    repeat the last warm-up frame
    warp    the last warm-up frame re-projected by the ego's logged motion (others stand still in the world)
    cv      constant velocity: road users seen during warm-up keep their last speed and heading
    ctrv    constant turn rate: ... keep their last speed and yaw rate (follows the ring's curve)
    ctrv_blur  ctrv, Gaussian-blurred (sigma per horizon step) and scaled (alpha per step), both tuned on TRAIN windows
            to minimise MSE. MSE rewards hedging (a blurry mean beats a sharp guess); this row hedges too, so "model
            beats ctrv_blur" means the model knows more than a kinematic guess plus the best fixed hedge.
    model   the checkpoint's open-loop prediction (z from the prior, dropout off), if --ckpt is given
For cv/ctrv only road users visible in the crop at some warm-up frame are drawn (what the model could know), at their
logged size, from their logged state at the last warm-up frame; yaw rate from the heading change over the last 0.4 s.
Runs on CPU (the model part uses a GPU if there is one). Writes <out>.csv (ctrv_blur's sigma/alpha in its columns).
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from fov_popin import crops, visible_mask
from roundabout_wm.baselines import extrapolate, yaw_rate
from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.render import car_corners, render_frame
from roundabout_wm.round_io import load_recording
from roundabout_wm.wm_eval import warp_previous

REPORT = (5, 10, 20)


def blur(x, sigma):
    """Gaussian blur of (N, H, W) images with a normalised kernel, sigma in pixels (0 = unchanged)."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k = torch.tensor(k / k.sum(), dtype=torch.float32)
    t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)[:, None]
    t = torch.nn.functional.conv2d(t, k.view(1, 1, 1, -1), padding=(0, r))
    t = torch.nn.functional.conv2d(t, k.view(1, 1, -1, 1), padding=(r, 0))
    return t[:, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=3000, help="random subset of windows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None, help="also score this world model on the same windows")
    p.add_argument("--out", default="baselines.csv")
    p.add_argument("--tune_split", default="train", help="windows used to tune ctrv_blur (never the scored split)")
    p.add_argument("--n_tune", type=int, default=1000, help="random tuning windows (0 = all)")
    p.add_argument("--blur_sigmas", default="0,0.5,1,1.5,2,3,4,6,8", help="grid of blur widths in pixels")
    a = p.parse_args()
    if a.tune_split == a.split:
        raise ValueError("tune ctrv_blur on a different split than the one being scored")
    cfg = Config()
    d, rc = cfg.data, cfg.render
    n, T, dt = d.n_cond, d.n_pred, d.dt
    ahead, behind, half = crops(cfg)["current"]

    splits = json.load(open(os.path.join(a.cache, "split.json")))
    model, stats = None, None
    if a.ckpt:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        s = torch.load(a.ckpt, map_location=dev)
        model = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        model.load_state_dict(s["model"]); stats = s["stats"]

    def windows(split_name, n_windows, seed):
        """Yields (window index, dataset item, ego states, target green (T,H,W), ctrv and cv renders)."""
        split = splits[split_name]
        ds = RounDWindows(a.cache, split, cfg, stats, n_pred=T)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(ds)) if n_windows <= 0 else np.sort(rng.choice(len(ds), min(n_windows, len(ds)),
                                                                           replace=False))
        print(f"{len(idx)} {split_name} windows from recordings {split}")
        by_rec, t0 = {}, time.time()
        for i in idx:
            by_rec.setdefault(int(ds.index[i][0]), []).append(int(i))
        for k_rec, wins in by_rec.items():
            rec_id = split[k_rec]
            rec = load_recording(a.data_dir, rec_id, step=d.step)
            tr = rec.tracks.join(rec.tracks_meta[["length", "width"]], on="trackId")
            frames = {int(f): g.set_index("trackId") for f, g in tr.groupby("frame", sort=False)}
            ego_frames = {int(t): g["frame"].to_numpy() for t, g in rec.tracks.groupby("trackId")}
            tracks_csv = pd.read_csv(os.path.join(a.cache, f"rec_{rec_id:02d}", "tracks.csv"))
            for i in wins:
                s_row = int(ds.index[i][1])
                row = tracks_csv[(tracks_csv.start <= s_row) & (s_row < tracks_csv.start + tracks_csv.length)].iloc[0]
                ego_id, off = int(row.trackId), s_row - int(row.start)
                fr = ego_frames[ego_id][off:off + n + T]
                item = ds[i]
                st = item["ego_state"].numpy().astype(np.float64)
                tgt = item["images"][n:, GREEN].numpy()                              # (T, 64, 64)
                L_e, W_e = rec.tracks_meta.at[ego_id, "length"], rec.tracks_meta.at[ego_id, "width"]
                ego_log = np.array([[frames[int(f)].at[ego_id, "x"], frames[int(f)].at[ego_id, "y"]]
                                    for f in fr[[0, -1]]])
                if not np.allclose(ego_log, st[[0, -1], :2], atol=1e-3):     # cache row <-> raw frame alignment
                    raise RuntimeError(f"window {i}: cached ego pose does not match the raw track at frames "
                                       f"{fr[[0, -1]]}")

                # road users visible in the crop at some warm-up frame (not the ego), with their state at frame n-1
                seen = set()
                for j in range(n):
                    g = frames[int(fr[j])].drop(index=ego_id, errors="ignore")
                    cor = car_corners(g["x"], g["y"], g["heading"], g["length"], g["width"])
                    vis = visible_mask(cor, st[j, 0], st[j, 1], st[j, 2], ahead, behind, half)
                    seen.update(g.index[vis].tolist())
                last = frames[int(fr[n - 1])]
                back = frames.get(int(fr[n - 3]), last)
                ids = [t for t in seen if t in last.index]
                g = last.loc[ids]
                h_back = np.array([back.at[t, "heading"] if t in back.index else last.at[t, "heading"] for t in ids])
                w = yaw_rate(g["heading"].to_numpy(), h_back, 2 * dt) if ids else np.zeros(0)
                kin = {}
                for name, turn in (("cv", False), ("ctrv", True)):
                    out = np.zeros_like(tgt)
                    for k in range(T):
                        xs, ys, hs = extrapolate(g["x"], g["y"], g["heading"], g["speed"], w, (k + 1) * dt, turn)
                        ego = dict(x=st[n + k, 0], y=st[n + k, 1], heading=st[n + k, 2], length=L_e, width=W_e)
                        others = dict(x=xs, y=ys, heading=hs, length=g["length"].to_numpy(),
                                      width=g["width"].to_numpy())
                        out[k] = render_frame(ego, others, None, rc)[1] / 255.0
                    kin[name] = out
                yield i, item, st, tgt, kin
            print(f"  rec {rec_id:02d}: {len(wins)} windows ({time.time() - t0:.0f} s)")

    # ---- tune ctrv_blur on train windows: per step k and sigma, sums for the closed-form best alpha -------------
    sigmas = [float(x) for x in a.blur_sigmas.split(",")]
    S_pp, S_py, S_yy = np.zeros((T, len(sigmas))), np.zeros((T, len(sigmas))), np.zeros(T)
    for _, _, _, tgt, kin in windows(a.tune_split, a.n_tune, a.seed + 1):
        for j, sg in enumerate(sigmas):
            pb = blur(kin["ctrv"], sg)
            S_pp[:, j] += (pb * pb).sum((1, 2)); S_py[:, j] += (pb * tgt).sum((1, 2))
        S_yy += (tgt * tgt).sum((1, 2))
    alpha_all = S_py / np.maximum(S_pp, 1e-12)                       # best scale for each (k, sigma)
    sse = S_yy[:, None] - S_py ** 2 / np.maximum(S_pp, 1e-12)        # train SSE with that scale
    best = sse.argmin(1)
    sigma_k = np.array([sigmas[j] for j in best]); alpha_k = alpha_all[np.arange(T), best]
    print("ctrv_blur tuned on " + a.tune_split + ": " + " | ".join(
        f"{(k) * dt:.0f} s sigma {sigma_k[k - 1]:.1f} px alpha {alpha_k[k - 1]:.2f}" for k in REPORT))

    # ---- score on the evaluation split ----------------------------------------------------------------------
    names = ["empty", "copy", "warp", "cv", "ctrv", "ctrv_blur"] + (["model"] if model else [])
    se = {k: np.zeros(T) for k in names}
    mass = {k: np.zeros(T) for k in names}
    mass_true = np.zeros(T)
    n_used = 0
    if model is not None:
        print(f"model {a.ckpt}")
    for i, item, st, tgt, kin in windows(a.split, a.n_windows, a.seed):
        preds = dict(kin)
        preds["empty"] = np.zeros_like(tgt)
        last_green = item["images"][n - 1, GREEN]
        preds["copy"] = np.broadcast_to(last_green.numpy(), tgt.shape)
        preds["warp"] = warp_previous(last_green[None, None], torch.tensor(st[n - 1, :3])[None],
                                      torch.tensor(st[n:, :3])[None], rc)[0, :, 0].numpy()
        preds["ctrv_blur"] = np.stack([alpha_k[k] * blur(kin["ctrv"][k:k + 1], sigma_k[k])[0] for k in range(T)])
        if model is not None:
            b = {key: v[None].to(next(model.parameters()).device) for key, v in item.items()}
            with torch.no_grad():
                torch.manual_seed(a.seed + i)
                preds["model"] = model(b["images"], b["actions"], b["speed"], T,
                                       z_source="prior")["green"][0, :, 0].float().cpu().numpy()
        for name in names:
            se[name] += ((preds[name] - tgt) ** 2).mean((1, 2))
            mass[name] += preds[name].sum((1, 2))
        mass_true += tgt.sum((1, 2))
        n_used += 1

    rows = []
    print(f"\n{n_used} windows | green-channel MSE (and MSE / empty), predicted / true mass")
    print(f"{'':10s}" + "".join(f"{f'{k * dt:.0f} s':>26s}" for k in REPORT))
    for name in names:
        cells = []
        for k in REPORT:
            m = se[name][k - 1] / n_used
            r = m / (se["empty"][k - 1] / n_used)
            ms = mass[name][k - 1] / max(mass_true[k - 1], 1e-9)
            cells.append(f"{m:9.5f} ({r:4.2f}) m{ms:4.2f}")
            tuned = name == "ctrv_blur"
            rows.append({"baseline": name, "step": k, "time_s": k * dt, "mse": m, "mse_over_empty": r, "mass": ms,
                         "sigma_px": sigma_k[k - 1] if tuned else "", "alpha": alpha_k[k - 1] if tuned else ""})
        print(f"{name:10s}" + "".join(f"{c:>26s}" for c in cells))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
