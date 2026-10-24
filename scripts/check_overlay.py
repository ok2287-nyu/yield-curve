"""Step 1 check: draw every vehicle trajectory of one recording on its background image.

Pass: cars follow the lanes and circulate counter-clockwise. Mirrored or shifted paths = wrong conversion.

    python scripts/check_overlay.py --data_dir /path/to/rounD/data --rec 17 --out overlay_17.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roundabout_wm.config import Config
from roundabout_wm.round_io import load_recording, world_to_background_px


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--rec", type=int, default=17)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = Config()

    rec = load_recording(a.data_dir, a.rec, step=1)
    m = rec.background_m_per_px(cfg.data.background_scale_down)
    img = plt.imread(rec.background_path)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    cls = rec.tracks_meta["class"]
    for t, g in rec.tracks.groupby("trackId"):
        col, row = world_to_background_px(g["x"], g["y"], m)
        color = "tab:red" if cls[t] == "car" else "tab:cyan"
        ax.plot(col, row, lw=0.4, alpha=0.6, color=color)
        ax.plot(col.iloc[-1] if hasattr(col, "iloc") else col[-1],
                row.iloc[-1] if hasattr(row, "iloc") else row[-1], ".", ms=2, color="yellow")
    ax.set_title(f"Recording {a.rec}: trajectories (red = cars, cyan = others, yellow = track end)")
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)

    col, row = world_to_background_px(rec.tracks["x"], rec.tracks["y"], m)
    outside = np.mean((col < 0) | (col >= img.shape[1]) | (row < 0) | (row >= img.shape[0]))
    print(f"background {img.shape[1]}x{img.shape[0]} px, {m:.4f} m/px; "
          f"{100 * outside:.2f}% of track points fall outside the image")

    out = a.out or f"overlay_{a.rec:02d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
