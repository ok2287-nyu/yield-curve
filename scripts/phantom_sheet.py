"""See the phantom cars: frames where the human coasted, we ask the model what happens if the ego brakes instead, and
the model draws road users that are not there. (CPU or GPU; images contain rounD: keep them on HPC / your machine.)

    python scripts/phantom_sheet.py --cache ../round_cache --data_dir ../round_data \
        --ckpt runs/ab_anchor/last.pt --csv runs/ab_anchor/safety_check.csv --top 8 --out_dir runs/phantom_sheets

Windows are chosen from safety_check.csv (same run): the human's plan is close to coasting (mean |accel| below
--coast), neither plan collides, ranked by how much phantom mass the model adds under brake compared with the
human's plan. Use --windows to pick windows yourself.

Each sheet, columns = last warm-up frame, then +0.4 .. +2.0 s. Red = road, blue = ego, green = other road users.
    real | human        the logged future (what happened)
    model | human       the model's prediction under the human's actions (mean of samples x masks)
    real | if braked    the logged road users seen from where the ego would be had it braked at -3 m/s^2
                        (exact if nobody reacts to the ego)
    model | if braked   the model's prediction under braking (mean)
    model | braked #1   one prior sample, one dropout mask (no averaging blur)
    phantom             model | if braked, with magenta where it predicts a road user (> --thresh) but none is
                        within 1.5 m in the real frame: invented traffic
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

from inspect_wm import compose, labelled_grid, to_img
from plan_rank import NAMES, score_window
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel


def pick_windows(csv, run, top, coast):
    d = pd.read_csv(csv)
    d = d[d.run == run] if "run" in d else d
    piv = d.pivot(index="window", columns="plan")
    acc_mean = np.sqrt(piv.acc2["log"] / 10)                     # rms of the human's accelerations over 2 s
    ok = (acc_mean < coast) & ~piv.crash["brake"] & ~piv.crash["log"] & (piv.first_hit["brake"].fillna("") != "rear")
    gain = (piv.phantom["brake"] - piv.phantom["log"])[ok].sort_values(ascending=False)
    print(f"{ok.sum()} coasting windows without collisions; top phantom gain under brake:")
    for w, g in gain.head(top).items():
        print(f"  window {w}: phantom brake {piv.phantom['brake'][w]:.2f} vs human {piv.phantom['log'][w]:.2f}, "
              f"human rms accel {acc_mean[w]:.2f} m/s^2")
    return [int(w) for w in gain.head(top).index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None, help="safety_check.csv to choose windows from")
    p.add_argument("--windows", type=int, nargs="*", default=None)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--coast", type=float, default=0.5, help="max rms accel of the human's plan, m/s^2")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/phantom_sheets")
    a = p.parse_args()
    cfg, cc = Config(), CostConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(a.ckpt, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    windows = a.windows or pick_windows(a.csv, run, a.top, a.coast)
    split = json.load(open(os.path.join(a.cache, "split.json")))["val"]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=10)
    os.makedirs(a.out_dir, exist_ok=True)
    T, n, steps = 10, cfg.data.n_cond, [1, 3, 5, 7, 9]               # step index k <-> +(k + 1) * 0.2 s
    ib, il = NAMES.index("brake"), NAMES.index("log")
    for w in windows:
        r = score_window(m, s, env, w, T, a.samples, a.masks, a.seed, cc, cfg, dev)
        if r is None:
            print(f"window {w}: recording ends inside 2 s, skipped"); continue
        ctx = r[1]
        poses = ctx["real"]["poses"]                                  # (P, T + 1, 3)
        last = torch.from_numpy(env.item["images"][n - 1].numpy()).float()[None]   # (1, 3, H, W) in [0, 1]

        def real_frames(pi):
            return torch.stack([torch.from_numpy(env.render(tuple(poses[pi, k + 1]), k + 1)).float() / 255
                                for k in steps])                      # (5, 3, H, W)

        def model_frames(pi, green):                                  # green (T, H, W) for plan pi
            rf = real_frames(pi)
            return compose(green[steps][:, None].cpu().float(), rf[:, :1], rf[:, 2:3])

        def row(frames):                                              # prepend the last warm-up frame
            return to_img(torch.cat([compose(last[:, 1:2], last[:, :1], last[:, 2:3]), frames]))

        G = ctx["green"].float()                                      # (M, S, P, T, H, W)
        Gm = G.mean((0, 1))
        rl, rb = real_frames(il), real_frames(ib)
        mb = model_frames(ib, Gm[ib])
        occ = F.max_pool2d((rb[:, 1:2] > 0.1).float(), 5, 1, 2)       # real road users, dilated 2 px
        ghost = (Gm[ib][steps][:, None].cpu() > a.thresh) & (occ < 0.5)
        ph = mb.clone()
        ph[:, 0] = torch.where(ghost[:, 0], torch.ones_like(ph[:, 0]), ph[:, 0])
        ph[:, 2] = torch.where(ghost[:, 0], torch.ones_like(ph[:, 2]), ph[:, 2])
        rows = [row(compose(rl[:, 1:2], rl[:, :1], rl[:, 2:3])), row(model_frames(il, Gm[il])),
                row(compose(rb[:, 1:2], rb[:, :1], rb[:, 2:3])), row(mb), row(model_frames(ib, G[0, 0, ib])),
                row(ph)]
        labels = ["real | human", "model | human", "real | if braked", "model | if braked", "model | braked #1",
                  "phantom (magenta)"]
        out = os.path.join(a.out_dir, f"phantom_{run}_w{w}.png")
        labelled_grid(rows, labels, [0.0] + [(k + 1) * cfg.data.dt for k in steps]).save(out)
        print(f"window {w}: phantom pixels at +0.4..2.0 s: {ghost.sum((1, 2, 3)).tolist()} -> {out}")


if __name__ == "__main__":
    main()
