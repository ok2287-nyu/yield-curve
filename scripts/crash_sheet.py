"""Draw crashing plans: what the model predicts vs what really happens, frame by frame. (CPU or GPU; images contain
rounD: keep them on HPC / your machine.)

    python scripts/crash_sheet.py --cache ../round_cache --data_dir ../round_data --ckpt runs/ab_anchor/last.pt \
        --exploit runs/ab_anchor/exploit_cases.csv --gain runs/ab_anchor/action_gain.csv --unexplained \
        --out_dir runs/crash_sheets

Cases come from exploit_cases.csv (same run): every plan whose first real collision is front or side.
--unexplained keeps only the cases without a lead gain in action_gain.csv (plan, 2 s): the crashes the
lead-mirroring join could not explain. --cases w:plan:track ... draws cases you name.

One sheet per case. Columns: last warm-up frame, then every 0.2 s step to +2.0 s; the step of the first real collision
is marked "HIT". Red = road, blue = ego, green = other road users.
    model | <plan>    the model's prediction under the crashing plan (mean of samples x masks)
    real | <plan>     the logged road users seen from where the plan takes the ego (replay: nobody reacts)
In both rows the car that is hit is outlined in yellow at its TRUE position, so in the model row you can see
whether the model puts it there, somewhere else, or nowhere.
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
from PIL import ImageDraw

from inspect_wm import LABEL_W, SCALE, compose, labelled_grid, to_img
from plan_rank import NAMES, score_window
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel

K3 = np.ones((3, 3), np.uint8)


def outline(fp):
    m = (fp > 0.1).astype(np.uint8)
    return (cv2.dilate(m, K3) > 0) & ~(m > 0)


def pick(exploit, gain, run, unexplained):
    """-> list of (window, plan, track, t_hit) from exploit_cases.csv."""
    e = pd.read_csv(exploit)
    e = e[e.run == run] if "run" in e else e
    if unexplained:
        g = pd.read_csv(gain)
        g = g[g.run == run] if "run" in g else g
        lead = g[(g.group == "lead") & np.isclose(g.t, 2.0)][["window", "plan", "gain"]]
        e = e.merge(lead, on=["window", "plan"], how="left")
        e = e[e.gain.isna()]
    print(f"{len(e)} cases:")
    print(e[["window", "plan", "kind", "cls", "track", "t_hit"]].to_string(index=False))
    return [(int(r.window), r.plan, int(r.track), float(r.t_hit)) for r in e.itertuples()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--exploit", default=None, help="exploit_cases.csv")
    p.add_argument("--gain", default=None, help="action_gain.csv (for --unexplained)")
    p.add_argument("--unexplained", action="store_true", help="only cases without a lead gain")
    p.add_argument("--cases", nargs="*", default=None, help="window:plan:track, e.g. 446:go:17")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/crash_sheets")
    a = p.parse_args()
    cfg, cc = Config(), CostConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(a.ckpt, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    if a.cases:
        chosen = [(int(c.split(":")[0]), c.split(":")[1], int(c.split(":")[2]), None) for c in a.cases]
    elif a.exploit:
        if a.unexplained and not a.gain:
            p.error("--unexplained needs --gain")
        chosen = pick(a.exploit, a.gain, run, a.unexplained)
    else:
        p.error("give --exploit or --cases")
    split = json.load(open(os.path.join(a.cache, "split.json")))["val"]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=10)
    os.makedirs(a.out_dir, exist_ok=True)
    T, n = 10, cfg.data.n_cond
    for w, plan, tid, t_hit in chosen:
        r = score_window(m, s, env, w, T, a.samples, a.masks, a.seed, cc, cfg, dev)
        if r is None:
            print(f"window {w}: recording ends inside 2 s, skipped"); continue
        ctx = r[1]
        pi = NAMES.index(plan)
        poses = ctx["real"]["poses"]                                    # (P, T + 1, 3)
        Tr = ctx["real"]["green"].shape[1]
        steps = list(range(min(T, Tr)))                                 # step k <-> +(k + 1) * 0.2 s
        if t_hit is None:                                               # first real hit of this track
            hits = [k for k in steps if any(h[0] == tid for h in ctx["real"]["hit_ids"][pi][k])]
            t_hit = (hits[0] + 1) * cfg.data.dt if hits else None
        last = torch.from_numpy(env.item["images"][n - 1].numpy()).float()[None]
        real = torch.stack([torch.from_numpy(env.render(tuple(poses[pi, k + 1]), k + 1)).float() / 255
                            for k in steps])                            # (K, 3, H, W)
        Gm = ctx["green"].float().mean((0, 1))[pi][steps].cpu()         # (K, H, W)
        model = compose(Gm[:, None], real[:, :1], real[:, 2:3])
        realc = compose(real[:, 1:2], real[:, :1], real[:, 2:3])
        for j, k in enumerate(steps):
            o = torch.from_numpy(outline(env.footprint(tid, poses[pi, k + 1], k + 1)))
            for img in (model, realc):
                img[j, 0][o], img[j, 1][o], img[j, 2][o] = 1.0, 1.0, 0.0
        first = compose(last[:, 1:2], last[:, :1], last[:, 2:3])
        rows = [to_img(torch.cat([first, model])), to_img(torch.cat([first, realc]))]
        times = [0.0] + [(k + 1) * cfg.data.dt for k in steps]
        grid = labelled_grid(rows, [f"model | {plan}", f"real | {plan}"], times)
        if t_hit is not None:                                           # mark the column of the first real hit
            j = int(round(t_hit / cfg.data.dt))                         # column index (0 = warm-up frame)
            cell = 64 * SCALE
            d = ImageDraw.Draw(grid)
            x0 = LABEL_W + j * cell
            d.rectangle([x0, 16, x0 + cell - 1, grid.height - 1], outline=(255, 60, 60), width=2)
            d.text((x0 + cell - 30, 2), "HIT", fill=(255, 80, 80))
        out = os.path.join(a.out_dir, f"crash_{run}_w{w}_{plan}.png")
        grid.save(out)
        print(f"window {w} {plan}, hit track {tid} at {t_hit} s -> {out}")


if __name__ == "__main__":
    main()
