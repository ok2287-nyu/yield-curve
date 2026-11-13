"""See action_gain.py's numbers as pictures: where does the model put the lead / follower when the ego changes its
plan? (CPU or GPU; images contain rounD: keep them on HPC / your machine.)

    python scripts/gain_sheet.py --cache ../round_cache --data_dir ../round_data --ckpt runs/ab_anchor/last.pt \
        --csv runs/ab_anchor/action_gain.csv --group follower --plan brake --top 6 --out_dir runs/gain_sheets

Windows come from action_gain.csv (same run, rows of --group under --plan at 2 s), sorted by gain; --order spread
(default) takes --top windows evenly from the lowest to the highest gain, low / high the extremes. --windows picks
windows yourself (the group's road user is then found as in action_gain.py).

Each sheet, columns = last warm-up frame, then +0.4 .. +2.0 s. Red = road, blue = ego, green = other road users.
    real | human          the logged future
    model | human         the model's prediction under the human's actions (mean of samples x masks)
    real | if <plan>      the logged road users seen from where the ego would be under <plan> (nobody reacts)
    model | if <plan>     the model's prediction under <plan> (mean)
    model | <plan> #1     one prior sample, one dropout mask
    where is it?          model | if <plan> with two outlines of the chosen road user:
                              yellow  = where it is if it does not react (replay; gain 1)
                              magenta = where it would be if glued to the ego (same place in the frame as under the
                                        human's plan; gain 0)
                          green on yellow: the model moves it as the ego's motion requires; green on magenta: the
                          model makes it copy the ego. For a follower the right answer lies between (it reacts, after
                          ~1 s); for a lead it is yellow.
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

from action_gain import GROUPS, groups
from inspect_wm import compose, labelled_grid, to_img
from plan_rank import NAMES, score_window
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel

K3 = np.ones((3, 3), np.uint8)


def pick(csv, run, group, plan, top, order):
    """-> [(window, trackId)] from action_gain.csv, and the csv rows of this run (for printing the gains)."""
    d = pd.read_csv(csv)
    d = d[d.run == run] if "run" in d else d
    sel = d[(d.group == group) & (d.plan == plan) & (np.isclose(d.t, 2.0))].sort_values("gain")
    if sel.empty:
        raise SystemExit(f"no {group} rows under {plan} at 2 s for run {run} in {csv}")
    n = len(sel)
    if order == "low":
        idx = range(min(top, n))
    elif order == "high":
        idx = range(max(0, n - top), n)
    else:
        idx = np.unique(np.linspace(0, n - 1, min(top, n)).round().astype(int))
    rows = sel.iloc[list(idx)]
    print(f"{n} {group} cases under {plan} at 2 s (median gain {sel.gain.median():.2f}); chosen:")
    return [(int(r.window), int(r.track)) for r in rows.itertuples()], d


def outline(fp):
    """(H, W) footprint in [0, 1] -> (H, W) bool boundary, 1 px."""
    m = (fp > 0.1).astype(np.uint8)
    return (cv2.dilate(m, K3) > 0) & ~(m > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None, help="action_gain.csv to choose windows from")
    p.add_argument("--windows", type=int, nargs="*", default=None)
    p.add_argument("--group", choices=GROUPS[:2], default="follower")
    p.add_argument("--plan", choices=NAMES[1:], default="brake")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--order", choices=["spread", "low", "high"], default="spread")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/gain_sheets")
    a = p.parse_args()
    if not a.windows and not a.csv:
        p.error("give --csv (action_gain.csv) or --windows")
    cfg, cc = Config(), CostConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(a.ckpt, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    if a.windows:
        chosen, d = [(w, None) for w in a.windows], None
    else:
        chosen, d = pick(a.csv, run, a.group, a.plan, a.top, a.order)
    split = json.load(open(os.path.join(a.cache, "split.json")))["val"]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, s["stats"], horizon=10)
    os.makedirs(a.out_dir, exist_ok=True)
    T, n, steps = 10, cfg.data.n_cond, [1, 3, 5, 7, 9]                 # step index k <-> +(k + 1) * 0.2 s
    ip, il = NAMES.index(a.plan), NAMES.index("log")
    for w, tid in chosen:
        r = score_window(m, s, env, w, T, a.samples, a.masks, a.seed, cc, cfg, dev)
        if r is None:
            print(f"window {w}: recording ends inside 2 s, skipped"); continue
        if tid is None:
            tid = next((t for t, g in groups(env, T).items() if g == a.group), None)
            if tid is None:
                print(f"window {w}: no {a.group}, skipped"); continue
        ctx = r[1]
        poses = ctx["real"]["poses"]                                    # (P, T + 1, 3)
        last = torch.from_numpy(env.item["images"][n - 1].numpy()).float()[None]

        def real_frames(pi):
            return torch.stack([torch.from_numpy(env.render(tuple(poses[pi, k + 1]), k + 1)).float() / 255
                                for k in steps])                        # (5, 3, H, W)

        def model_frames(pi, green):
            rf = real_frames(pi)
            return compose(green[steps][:, None].cpu().float(), rf[:, :1], rf[:, 2:3])

        def row(frames):
            return to_img(torch.cat([compose(last[:, 1:2], last[:, :1], last[:, 2:3]), frames]))

        G = ctx["green"].float()                                        # (M, S, P, T, H, W)
        Gm = G.mean((0, 1))
        rl, rp = real_frames(il), real_frames(ip)
        mp = model_frames(ip, Gm[ip])
        where = mp.clone()
        for j, k in enumerate(steps):
            glued = torch.from_numpy(outline(env.footprint(tid, poses[il, k + 1], k + 1)))
            free = torch.from_numpy(outline(env.footprint(tid, poses[ip, k + 1], k + 1)))
            for c, on in ((0, glued), (2, glued), (0, free), (1, free)):
                where[j, c][on] = 1.0
            where[j, 2][free] = 0.0                                     # yellow wins where the outlines touch
        rows = [row(compose(rl[:, 1:2], rl[:, :1], rl[:, 2:3])), row(model_frames(il, Gm[il])),
                row(compose(rp[:, 1:2], rp[:, :1], rp[:, 2:3])), row(mp), row(model_frames(ip, G[0, 0, ip])),
                row(where)]
        labels = ["real | human", "model | human", f"real | if {a.plan}", f"model | if {a.plan}",
                  f"model | {a.plan} #1", "where is it?"]
        out = os.path.join(a.out_dir, f"gain_{run}_{a.group}_{a.plan}_w{w}.png")
        labelled_grid(rows, labels, [0.0] + [(k + 1) * cfg.data.dt for k in steps]).save(out)
        g = ""
        if d is not None:
            gg = d[(d.window == w) & (d.track == tid) & (d.plan == a.plan)].set_index("t").gain
            g = "  gain " + ", ".join(f"{t:.0f} s {v:.2f}" for t, v in gg.items())
        print(f"window {w}, {a.group} track {tid}:{g} -> {out}")


if __name__ == "__main__":
    main()
