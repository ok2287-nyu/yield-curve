"""Imagined driving without a GUI: compare ego plans from the same real moment as one side-by-side GIF.

    python scripts/drive_wm.py --ckpt runs/cnn_det/best.pt --cache ../round_cache --data_dir ../round_data \
        --scene 0 --plans log brake go --seconds 4

Panels: the real log | one panel per plan. Each plan starts from the same 2 s real warm-up, then the ego follows
the plan (the logged path's curvature, so it stays on its route) and the model imagines the other road users.
    log      the human's own accelerations (the panel that should look most like the real log)
    brake    -3 m/s^2 until stopped (yield)          go      +1.5 m/s^2 (push in)
    coast    0 m/s^2                                  a number, e.g. -1.2 or 0.8, for a constant acceleration
Past 4 s (--seconds > 4) the model is beyond its training horizon. For the interactive version, in Jupyter:
    from roundabout_wm.sim import play; play("runs/cnn_det/best.pt", "../round_cache", "../round_data")
Frames are rendered rounD data: keep the GIF on HPC / your own machine.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from roundabout_wm.sim import ImaginedDrive, to_rgb

NAMED = {"log": None, "brake": -3.0, "go": 1.5, "coast": 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True, help="the rounD data folder (backgrounds + recordingMeta)")
    p.add_argument("--split", default="val")
    p.add_argument("--scene", type=int, default=0, help="window index in the split (same numbering as inspect_wm)")
    p.add_argument("--plans", nargs="+", default=["log", "brake", "go"])
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sim = ImaginedDrive.load(a.ckpt, a.cache, a.data_dir, a.split)
    steps = int(round(a.seconds / sim.cfg.data.dt))
    panels = {"real log": []}
    sim.reset(a.scene, seed=a.seed)
    warm = list(sim.history)
    sim.save("start")
    for plan in a.plans:
        acc = NAMED[plan] if plan in NAMED else float(plan)
        sim.restore("start"); sim.reseed(a.seed)
        frames = []
        for _ in range(steps):
            frames.append(sim.step(accel=acc, curvature=None))
        panels[f"{plan} ({'human' if acc is None else f'{acc:+.1f}'})"] = (frames, sim.state[3])
    sim.restore("start")
    truth = []
    for k in range(steps):
        sim.t = k + 1
        truth.append(sim.truth())
    panels["real log"] = (truth, None)

    S = 5
    names = list(panels)
    gif = []
    for t in range(len(warm) + steps):
        tiles = []
        for name in names:
            if t < len(warm):
                fr = warm[t]
            else:
                fr = panels[name][0][t - len(warm)]
            tiles.append(to_rgb(fr, S) if fr is not None else np.zeros((64 * S, 64 * S, 3), np.uint8))
        row = np.concatenate([np.pad(x, ((0, 0), (0, 4), (0, 0))) for x in tiles], 1)
        im = Image.new("RGB", (row.shape[1], row.shape[0] + 18), (20, 20, 20))
        im.paste(Image.fromarray(row), (0, 18))
        d = ImageDraw.Draw(im)
        for j, name in enumerate(names):
            d.text((j * (64 * S + 4) + 4, 3), name, fill=(230, 230, 230))
        tt = (t - len(warm) + 1) * sim.cfg.data.dt
        label = "real warm-up" if t < len(warm) else f"imagined +{tt:.1f}s" + ("  (beyond 4 s training horizon)" if tt > 4.0001 else "")
        d.text((im.width - 260, 3), label, fill=(255, 200, 80) if t >= len(warm) else (160, 160, 160))
        gif.append(im)
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"drive_scene{a.scene}_{'_'.join(a.plans)}.gif")
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=200, loop=0)
    for name in names[1:]:
        print(f"  {name:22s} final speed {panels[name][1]:.1f} m/s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
