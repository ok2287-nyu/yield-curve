"""Step 2 check: do the extracted actions reproduce the recorded paths?

For every car in a recording: extract (acceleration, curvature) at 5 Hz, integrate them open-loop from the
first state, and measure the position error. Short horizons matter most (the world model predicts 3 s);
whole-track errors show drift.

    python scripts/check_actions.py --data_dir /path/to/data --rec 17 --track 30
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
from roundabout_wm.kinematics import extract_actions, rollout_np
from roundabout_wm.round_io import load_recording


def reconstruct(g, d):
    st = g[["x", "y", "heading", "speed"]].to_numpy()
    acts = extract_actions(st[:, 2], st[:, 3], d.dt, d.v_min_curvature, d.accel_clip, d.curvature_clip)
    return st, acts, rollout_np(st[0], acts, d.dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--rec", type=int, default=17)
    p.add_argument("--track", type=int, default=30)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = Config(); d = cfg.data

    rec = load_recording(a.data_dir, a.rec, step=d.step)
    cars = rec.tracks_meta.index[rec.tracks_meta["class"] == "car"]

    # Error after a fixed horizon, starting from every point of every car track.
    horizon = d.n_pred                      # 15 steps = 3 s
    errs_h, errs_full, clipped = [], [], 0
    for t, g in rec.tracks[rec.tracks["trackId"].isin(cars)].groupby("trackId"):
        st, acts, _ = reconstruct(g, d)
        raw_a = np.diff(st[:, 3]) / d.dt
        clipped += int(np.sum(np.abs(raw_a) > d.accel_clip))
        if len(st) > horizon:
            for s in range(0, len(st) - horizon, 5):
                rr = rollout_np(st[s], acts[s:s + horizon], d.dt)
                errs_h.append(np.hypot(*(rr[-1, :2] - st[s + horizon, :2])))
        full = rollout_np(st[0], acts, d.dt)
        errs_full.append(np.hypot(*(full[-1, :2] - st[-1, :2]).T))
    errs_h, errs_full = np.array(errs_h), np.array(errs_full)
    print(f"cars: {len(errs_full)} | 3 s open-loop position error: median {np.median(errs_h):.2f} m, "
          f"95th pct {np.percentile(errs_h, 95):.2f} m, max {errs_h.max():.2f} m")
    print(f"whole-track end error: median {np.median(errs_full):.2f} m, 95th pct {np.percentile(errs_full, 95):.2f} m")
    print(f"accelerations clipped: {clipped}")

    # Plot one track: recorded vs reconstructed, plus the actions.
    g = rec.tracks[rec.tracks["trackId"] == a.track]
    st, acts, rr = reconstruct(g, d)
    tt = np.arange(len(st)) * d.dt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(st[:, 0], st[:, 1], "k-", lw=2, label="recorded")
    ax[0].plot(rr[:, 0], rr[:, 1], "r--", lw=1.5, label="integrated actions")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title(f"track {a.track}: path (m)")
    ax[1].plot(tt[:-1], acts[:, 0], label="acceleration (m/s²)")
    ax[1].plot(tt, st[:, 3], label="speed (m/s)")
    ax[1].legend(); ax[1].set_xlabel("time (s)")
    ax[2].plot(tt[:-1], acts[:, 1], label="curvature (1/m)")
    ax[2].axhline(0, color="gray", lw=0.5); ax[2].legend(); ax[2].set_xlabel("time (s)")
    out = a.out or f"actions_{a.rec:02d}_{a.track}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"track {a.track}: end error {np.hypot(*(rr[-1, :2] - st[-1, :2])):.2f} m; wrote {out}")


if __name__ == "__main__":
    main()
