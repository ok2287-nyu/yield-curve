"""One sheet per scene: the truth, the deterministic model, then each stochastic model's posterior and prior samples.

    python scripts/compare_models.py --cache ../round_cache \
        --det runs/cnn_det/best.pt --stoch runs/ws_both/best.pt runs/ws_ol50/best.pt runs/ws_b1e-3/best.pt \
        --scenes 0 72 145 218 290 363 --k 4

Rows (columns every 0.4 s of the 4 s horizon):
    truth                         the real future
    deterministic                 the stage-1 model (the conditional mean)
    <run> posterior               z from the posterior (it saw the next frame): what z CAN carry
    <run> prior #1..#k            open-loop samples: what planning would see
    <run> spread                  per-pixel std of the k samples (white = they disagree; same grey scale for all runs)
Fading shows as cars that are in the posterior row but dissolve across a prior row; diversity as prior rows that
differ from each other (and a non-black spread row). Prints, per scene and model, at 1 s / 4 s: mass ratio, sample
spread, and the MSE of the mean of the k samples and of the best of the k.
Writes <out_dir>/compare_<scene>.png. Rendered rounD data: keep it on HPC / your machine.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from inspect_wm import compose, labelled_grid, to_img
from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel


def load(path, dev):
    s = torch.load(path, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    return m, s["stats"], s.get("step", 0)


def name_of(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--det", required=True, help="deterministic (stage-1) checkpoint")
    p.add_argument("--stoch", nargs="+", required=True, help="stochastic checkpoints to compare")
    p.add_argument("--scenes", type=int, nargs="+", default=[0, 72, 145, 218, 290, 363])
    p.add_argument("--k", type=int, default=4, help="prior samples per stochastic model")
    p.add_argument("--split", default="val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/compare")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    os.makedirs(a.out_dir, exist_ok=True)

    det, stats, det_step = load(a.det, dev)
    stochs = [(name_of(pth),) + load(pth, dev) for pth in a.stoch]
    for name, _, st, _ in stochs:
        if st != stats:
            raise ValueError(f"{name} was trained with different normalisation stats than the deterministic model")
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    ds = RounDWindows(a.cache, split, cfg, stats, n_pred=T)
    print(f"deterministic {a.det} (step {det_step}); stochastic: "
          + ", ".join(f"{nm} (step {stp})" for nm, _, _, stp in stochs))

    cols = list(range(1, T, 2))                                            # every 0.4 s: steps 2, 4, ..., 20
    col_times = [(k + 1) * dt for k in cols]
    sel = lambda x: x[cols]
    report = {}
    for idx in a.scenes:
        item = ds[idx]
        b = {k: v[None].to(dev) for k, v in item.items()}
        imgs, act, spd = b["images"], b["actions"], b["speed"]
        truth = item["images"][n:n + T]
        tg, road, ego = truth[:, GREEN:GREEN + 1], truth[:, 0:1], truth[:, 2:3]
        rows, labels = [to_img(sel(compose(tg, road, ego)))], ["truth"]
        g_det = det(imgs, act, spd, T)["green"][0].float().cpu()
        rows.append(to_img(sel(compose(g_det, road, ego)))); labels.append("deterministic")

        def numbers(samples):                                              # samples (K, T, 1, H, W)
            mse = ((samples - tg[None]) ** 2).mean((2, 3, 4))              # (K, T)
            mean_mse = ((samples.mean(0) - tg) ** 2).mean((1, 2, 3))
            m = samples.sum((2, 3, 4)).mean(0) / tg.sum((1, 2, 3)).clamp_min(1e-6)
            spread = samples.std(0).mean((1, 2, 3)) if samples.shape[0] > 1 else torch.zeros(T)
            return {k: (m[k - 1].item(), spread[k - 1].item(), mean_mse[k - 1].item(), mse[:, k - 1].min().item())
                    for k in (5, 20)}
        report[(idx, "deterministic")] = numbers(g_det[None])

        for name, m, _, _ in stochs:
            torch.manual_seed(a.seed)
            post = m(imgs, act, spd, T, z_source="post")["green"][0].float().cpu()
            rows.append(to_img(sel(compose(post, road, ego)))); labels.append(f"{name} post")
            samples = []
            for j in range(a.k):
                torch.manual_seed(a.seed + 1 + j)
                g = m(imgs, act, spd, T, z_source="prior")["green"][0].float().cpu()
                samples.append(g)
                rows.append(to_img(sel(compose(g, road, ego)))); labels.append(f"{name} prior #{j + 1}")
            S = torch.stack(samples)
            sp = (S.std(0) / 0.25).clamp(0, 1).expand(-1, 3, -1, -1)       # fixed scale: std 0.25 = white
            rows.append(to_img(sel(sp))); labels.append(f"{name} spread")
            report[(idx, name)] = numbers(S)
        labelled_grid(rows, labels, col_times).save(os.path.join(a.out_dir, f"compare_{idx}.png"))

    print(f"\nper scene: mass ratio / spread / MSE of mean-of-{a.k} / MSE of best-of-{a.k}   at 1 s | 4 s")
    for idx in a.scenes:
        print(f"scene {idx}")
        for nm in ["deterministic"] + [s[0] for s in stochs]:
            r = report[(idx, nm)]
            cell = lambda k: f"m{r[k][0]:4.2f} s{r[k][1]:.4f} mean {r[k][2]:.4f} best {r[k][3]:.4f}"
            print(f"  {nm:14s} 1 s: {cell(5)} | 4 s: {cell(20)}")
    print(f"wrote {len(a.scenes)} sheets to {a.out_dir}/")


if __name__ == "__main__":
    main()
