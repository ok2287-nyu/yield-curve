"""Step 4 check: render one ego track as an animated GIF, next to its position on the aerial image.

Pass: the road rotates smoothly as the ego turns, the ego (blue) stays fixed pointing up,
and other cars (green) stay in their lanes.

    python scripts/render_track.py --data_dir /path/to/data --rec 17 --track 30
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import imageio
import numpy as np

from roundabout_wm.build import _frame_lookup
from roundabout_wm.config import Config
from roundabout_wm.render import RoadMap, render_frame
from roundabout_wm.round_io import load_recording, world_to_background_px


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--rec", type=int, default=17)
    p.add_argument("--track", type=int, default=30)
    p.add_argument("--scale", type=int, default=6, help="upscale factor for viewing")
    p.add_argument("--hires", type=int, default=0,
                   help="render at this many pixels instead of the training size (e.g. 256) for presentations; "
                        "the field of view stays 48 m, only the pixels get finer")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = Config(); d, rc = cfg.data, cfg.render
    if a.hires:
        k = a.hires / rc.size
        rc = dataclasses.replace(rc, size=a.hires, m_per_px=rc.m_per_px / k, ego_row=round(rc.ego_row * k))
        a.scale = max(1, round(384 / a.hires))

    rec = load_recording(a.data_dir, a.rec, step=d.step)
    m = rec.background_m_per_px(d.background_scale_down)
    road = RoadMap(rec.background_path, m, rc)
    lookup = _frame_lookup(rec)
    bg = cv2.cvtColor(cv2.imread(rec.background_path), cv2.COLOR_BGR2RGB)
    L, W = rec.tracks_meta.at[a.track, "length"], rec.tracks_meta.at[a.track, "width"]

    g = rec.tracks[rec.tracks["trackId"] == a.track]
    out_frames = []
    view = rc.size * a.scale
    for _, r in g.iterrows():
        o = lookup[int(r.frame)]
        mask = o["trackId"] != a.track
        others = {k: o[k][mask] for k in ("x", "y", "heading", "length", "width")}
        ego = {"x": r.x, "y": r.y, "heading": r.heading, "length": L, "width": W}
        img = render_frame(ego, others, road, rc).transpose(1, 2, 0)        # H, W, 3 (R, G, B)
        img = cv2.resize(img, (view, view), interpolation=cv2.INTER_NEAREST)

        # global context: aerial image with the ego marked
        f = view / bg.shape[0]
        ctx = cv2.resize(bg, (int(bg.shape[1] * f), view), interpolation=cv2.INTER_AREA)
        col, row = world_to_background_px(r.x, r.y, m)
        c0 = np.array([col * f, row * f])
        tip = c0 + 14 * np.array([np.cos(r.heading), -np.sin(r.heading)])   # image rows point down
        cv2.circle(ctx, tuple(int(v) for v in c0), 7, (255, 60, 255), 2)
        cv2.arrowedLine(ctx, tuple(int(v) for v in c0), tuple(int(v) for v in tip), (255, 60, 255), 2, tipLength=0.4)
        frame = np.concatenate([img, ctx], axis=1)
        cv2.putText(frame, f"t={(r.frame - g.frame.iloc[0]) / d.source_hz:4.1f}s v={r.speed:4.1f}m/s",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out_frames.append(frame)

    out = a.out or f"render_{a.rec:02d}_{a.track}.gif"
    imageio.mimsave(out, out_frames, duration=d.dt, loop=0)
    print(f"{len(out_frames)} frames -> {out}")


if __name__ == "__main__":
    main()
