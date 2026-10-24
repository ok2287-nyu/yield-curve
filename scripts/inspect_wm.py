"""Look at what a trained world model actually predicts, to decide what to change.

    python scripts/inspect_wm.py --ckpt runs/cnn_b1e-4/best.pt --cache $C
    python scripts/inspect_wm.py --ckpt runs/cnn_b1e-4/best.pt --cache $C --scenes 120 4410 9001   # chosen windows

For each scene (a validation window with cars in its future) it writes, into <run>/inspect_<step>/:
    scene_<idx>.png   one row per source, one column per 0.4 s of the 4 s horizon:
        truth            the real future
        posterior        z from the posterior (it has seen the next frame): the best the decoder can do
        prior #1..#K     open-loop samples, z from the learned prior (what planning will see)
        prior mean       mean of the K samples
        prior spread     per-pixel std of the K samples (white = samples disagree)
        brake / speed up same scene, ego acceleration replaced by -3 / +1.5 m/s^2 (speeds re-integrated);
                         green only, because the road for a counterfactual ego path is not rendered
    scene_<idx>.gif   the same, animated: 2 s of real warm-up, then the 4 s prediction
    summary.csv       per scene and step: MSE of posterior / prior mean / best-of-K, sample spread, mass ratio

How to read it (which failure you are looking at):
    posterior blurry too                      -> decoder/z capacity problem (z carries too little)
    posterior sharp, prior samples identical  -> z is ignored when it comes from the prior
    posterior sharp, prior samples differ but are wrong   -> the prior cannot produce the right z
    prior samples start sharp and fade with the horizon  -> h drifts in long open-loop runs (exposure bias)
    brake and speed-up rows identical         -> the model ignores the ego's actions

Images are rendered from rounD data: keep them on HPC / your machine (git-ignored), never in the repo.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.wm_eval import counterfactual_accel

SCALE = 3          # pixel upscaling of the 64x64 frames
LABEL_W = 110      # width of the row-label margin (pixels)


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="runs/<run>/best.pt or last.pt")
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", default=None, help="default: <ckpt dir>/inspect_<step>")
    p.add_argument("--n_scenes", type=int, default=6)
    p.add_argument("--scenes", type=int, nargs="*", default=None, help="window indices to show instead")
    p.add_argument("--k", type=int, default=6, help="prior samples per scene")
    p.add_argument("--min_cars", type=float, default=30.0,
                   help="only auto-pick scenes whose 4 s future holds at least this much green mass (px)")
    p.add_argument("--brake", type=float, default=-3.0, help="counterfactual acceleration, m/s^2")
    p.add_argument("--speedup", type=float, default=1.5, help="counterfactual acceleration, m/s^2")
    p.add_argument("--no_gif", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def compose(green, road=None, ego=None):
    """(T, 1, H, W) green [+ red road, blue ego] -> (T, 3, H, W) RGB in [0, 1]."""
    T, _, H, W = green.shape
    out = torch.zeros(T, 3, H, W)
    if road is not None:
        out[:, 0] = road[:, 0] * 0.6            # dimmed so the green stands out
    out[:, 1] = green[:, 0]
    if ego is not None:
        out[:, 2] = ego[:, 0]
    return out.clamp(0, 1)


def to_img(frames_rgb):
    """(T, 3, H, W) -> one PIL row, frames side by side, upscaled."""
    row = torch.cat(list(frames_rgb), 2)                                   # (3, H, T*W)
    arr = (row.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def labelled_grid(rows, labels, col_times):
    w = LABEL_W + rows[0].width
    h_head = 16
    grid = Image.new("RGB", (w, h_head + sum(r.height for r in rows)), (20, 20, 20))
    d = ImageDraw.Draw(grid)
    cell = rows[0].width // len(col_times)
    for j, t in enumerate(col_times):
        d.text((LABEL_W + j * cell + 4, 2), f"+{t:.1f}s", fill=(220, 220, 220))
    y = h_head
    for r, lab in zip(rows, labels):
        grid.paste(r, (LABEL_W, y))
        d.text((6, y + r.height // 2 - 6), lab, fill=(230, 230, 230))
        y += r.height
    return grid


@torch.no_grad()
def main():
    a = args_()
    cfg = Config()
    n, dt = cfg.data.n_cond, cfg.data.dt
    s = torch.load(a.ckpt, map_location="cpu")
    wm_cfg = WMConfig(**s["wm_cfg"])
    stats = s["stats"]
    model = WorldModel(wm_cfg)
    model.load_state_dict(s["model"])
    model.eval()
    step = s.get("step", 0)
    out_dir = a.out or os.path.join(os.path.dirname(a.ckpt), f"inspect_{step:06d}")
    os.makedirs(out_dir, exist_ok=True)

    split = json.load(open(os.path.join(a.cache, "split.json")))
    n_pred = 20
    ds = RounDWindows(a.cache, split[a.split], cfg, stats, n_pred=n_pred)

    if a.scenes:
        scene_ids = a.scenes
    else:                                       # evenly spread over the split, skipping near-empty futures
        scene_ids, cands = [], np.linspace(0, len(ds) - 1, 40 * a.n_scenes).astype(int)
        for c in cands:
            if ds[int(c)]["images"][n:, GREEN].sum() / n_pred >= a.min_cars:
                scene_ids.append(int(c))
            if len(scene_ids) == a.n_scenes:
                break
    print(f"{a.ckpt} (step {step}) | {a.split} scenes {scene_ids} -> {out_dir}")

    col_steps = list(range(1, n_pred, 2))                                  # every 0.4 s: steps 2, 4, ..., 20
    col_times = [(k + 1) * dt for k in col_steps]
    rows_csv = []
    for idx in scene_ids:
        item = ds[idx]
        b = {k: v[None] for k, v in item.items()}
        imgs, act, spd, st = b["images"], b["actions"], b["speed"], b["ego_state"]
        truth = imgs[0, n:n + n_pred]
        tg = truth[:, GREEN:GREEN + 1]
        road, ego = truth[:, 0:1], truth[:, 2:3]

        torch.manual_seed(a.seed)
        post = model(imgs, act, spd, n_pred, z_source="post")["green"][0].float()
        priors = []
        for j in range(a.k):
            torch.manual_seed(a.seed + 1 + j)
            priors.append(model(imgs, act, spd, n_pred, z_source="prior")["green"][0].float())
        P = torch.stack(priors)                                            # (K, T, 1, H, W)
        mean, spread = P.mean(0), P.std(0)
        cf = {}
        for name, acc in (("brake", a.brake), ("speed up", a.speedup)):
            act_c, spd_c = counterfactual_accel(act, st, n, stats, dt, acc)
            torch.manual_seed(a.seed + 1)                                  # same noise as prior #1
            cf[name] = model(imgs, act_c, spd_c, n_pred, z_source="prior")["green"][0].float()

        # ---- numbers per step ------------------------------------------------------------------------------
        mse = lambda x: ((x - tg) ** 2).mean((1, 2, 3))
        per_sample = ((P - tg[None]) ** 2).mean((2, 3, 4))                 # (K, T)
        true_mass = tg.sum((1, 2, 3)).clamp_min(1e-6)
        for t in range(n_pred):
            rows_csv.append({"scene": idx, "step": t + 1, "time_s": round((t + 1) * dt, 1),
                             "mse_post": mse(post)[t].item(), "mse_prior_mean": mse(mean)[t].item(),
                             "mse_best_of_k": per_sample[:, t].min().item(),
                             "mse_prior_avg": per_sample[:, t].mean().item(),
                             "spread": spread[t].mean().item(),
                             "mass_prior": (P[:, t].sum((1, 2, 3)).mean() / true_mass[t]).item(),
                             "mass_post": (post[t].sum() / true_mass[t]).item(),
                             "cf_diff": (cf["brake"][t] - cf["speed up"][t]).abs().mean().item()})

        # ---- picture -----------------------------------------------------------------------------------------
        sel = lambda x: x[col_steps]
        rows = [to_img(sel(compose(tg, road, ego))), to_img(sel(compose(post, road, ego)))]
        labels = ["truth", "posterior"]
        for j in range(a.k):
            rows.append(to_img(sel(compose(P[j], road, ego)))); labels.append(f"prior #{j + 1}")
        rows.append(to_img(sel(compose(mean, road, ego)))); labels.append("prior mean")
        sp = (spread / spread.max().clamp_min(1e-6)).expand(-1, 3, -1, -1)
        rows.append(to_img(sel(sp))); labels.append("prior spread")
        for name in ("brake", "speed up"):
            rows.append(to_img(sel(compose(cf[name], None, ego)))); labels.append(name)
        labelled_grid(rows, labels, col_times).save(os.path.join(out_dir, f"scene_{idx}.png"))

        if not a.no_gif:                                                   # warm-up (real) then prediction
            w = imgs[0, :n]
            warm = compose(w[:, GREEN:GREEN + 1], w[:, 0:1], w[:, 2:3])            # same colours as predictions
            panels = [("truth", torch.cat([warm, compose(tg, road, ego)])),
                      ("posterior", torch.cat([warm, compose(post, road, ego)])),
                      ("prior #1", torch.cat([warm, compose(P[0], road, ego)])),
                      ("prior mean", torch.cat([warm, compose(mean, road, ego)])),
                      ("spread", torch.cat([torch.zeros_like(warm), sp]))]
            frames = []
            for t in range(n + n_pred):
                row = to_img(torch.stack([p[t] for _, p in panels]))
                fr = Image.new("RGB", (row.width, row.height + 16), (20, 20, 20))
                fr.paste(row, (0, 16))
                d = ImageDraw.Draw(fr)
                for j, (name, _) in enumerate(panels):
                    d.text((j * 64 * SCALE + 4, 2), name, fill=(230, 230, 230))
                tt = (t - n + 1) * dt
                d.text((fr.width - 80, 2), f"{'warm-up' if t < n else 'pred'} {tt:+.1f}s",
                       fill=(255, 200, 80) if t >= n else (160, 160, 160))
                frames.append(fr)
            frames[0].save(os.path.join(out_dir, f"scene_{idx}.gif"), save_all=True, append_images=frames[1:],
                           duration=200, loop=0)
        r1 = [r for r in rows_csv if r["scene"] == idx]
        at = lambda key, k: r1[k - 1][key]
        print(f"  scene {idx}: mse post/prior-mean/best-of-{a.k} @4s "
              f"{at('mse_post', 20):.4f}/{at('mse_prior_mean', 20):.4f}/{at('mse_best_of_k', 20):.4f} | "
              f"spread @1s/4s {at('spread', 5):.4f}/{at('spread', 20):.4f} | "
              f"mass prior @1s/4s {at('mass_prior', 5):.2f}/{at('mass_prior', 20):.2f} | "
              f"brake-vs-speedup @4s {at('cf_diff', 20):.4f}")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader(); w.writerows(rows_csv)
    print(f"wrote {len(scene_ids)} scenes to {out_dir}")


if __name__ == "__main__":
    main()
