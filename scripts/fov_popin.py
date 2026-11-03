"""Field of view: how often does a road user the model could never have seen enter the crop during the rollout?

    python scripts/fov_popin.py --data_dir $ROOT/round_data                  # all Neuweiler recordings (~5-15 min CPU)
    python scripts/fov_popin.py --data_dir $ROOT/round_data --recs 18 19 20  # just the val recordings

Uses the same ego windows as training (class car, cached tracks >= 25 frames, windows of n_cond + n_pred = 30
frames at 5 Hz starting every 5 frames) and raw rounD tracks, so it needs no cache and no GPU.

A road user is VISIBLE at a frame if any corner of its box lies inside the crop (the renderer draws partial boxes).
A POP-IN at rollout step k is a road user visible at frame n_cond-1+k that was not visible at ANY warm-up frame
(0 .. n_cond-1): the model had no way to know it exists. Reported at 1 / 2 / 4 s (k = 5 / 10 / 20):
    windows_with_popin   share of windows with >= 1 pop-in at step k
    popin_area_share     share of the visible box area at step k that belongs to pop-ins (box area L*W, not clipped
                         to the crop: a rough stand-in for the share of green-channel mass)
    close_popin          share of windows where a pop-in comes within --close_m of the ego centre by step k
                         (the ones that matter for yielding and collisions)
for three crop geometries:
    current      64 px x 0.75 m, ego at row 43   -> 32.6 m ahead, 15.4 m behind, 24 m each side
    ego_lower    64 px x 0.75 m, ego at row 55   -> 41.6 m ahead,  6.4 m behind, 24 m each side
    crop_96m     128 px x 0.75 m (or 64 px x 1.5 m), ego at the same relative row -> 65 m ahead, 31 m behind, 48 m
Also prints pop-ins at 4 s by road-user class, and writes everything to <out>.csv.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from roundabout_wm.build import eligible_ego_tracks
from roundabout_wm.config import Config
from roundabout_wm.render import car_corners
from roundabout_wm.round_io import list_recordings, load_recording

REPORT = (5, 10, 20)


def crops(cfg):
    """name -> (ahead, behind, half_width) in metres, from the renderer's conventions (pixel centres at integers)."""
    rc = cfg.render
    m, S, r = rc.m_per_px, rc.size, rc.ego_row

    def geom(size, row, mpp):
        return ((row + 0.5) * mpp, (size - row - 0.5) * mpp, size / 2 * mpp)
    return {"current": geom(S, r, m),
            "ego_lower": geom(S, 55, m),
            "crop_96m": geom(2 * S, 2 * r + 0.5, m)}


def visible_mask(corners, ex, ey, eh, ahead, behind, half):
    """corners (N, 4, 2) world -> (N,) bool: any corner inside the ego-centric crop rectangle."""
    d = corners - np.array([ex, ey])
    c, s = np.cos(eh), np.sin(eh)
    fwd = d[..., 0] * c + d[..., 1] * s
    left = -d[..., 0] * s + d[..., 1] * c
    inside = (fwd <= ahead) & (fwd >= -behind) & (np.abs(left) <= half)
    return inside.any(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--recs", type=int, nargs="*", default=None)
    p.add_argument("--close_m", type=float, default=15.0, help="'close' pop-in: within this distance of the ego")
    p.add_argument("--out", default="fov_popin.csv")
    a = p.parse_args()
    cfg = Config()
    d = cfg.data
    n, L, stride = d.n_cond, d.n_cond + d.n_pred, d.window_stride
    geoms = crops(cfg)
    for g, (ah, be, hw) in geoms.items():
        print(f"{g:10s} ahead {ah:5.1f} m  behind {be:5.1f} m  each side {hw:5.1f} m")
    recs = a.recs or list_recordings(a.data_dir, d.location_id)

    # accumulators: per geometry and step k
    acc = {g: {k: {"win": 0, "popin": 0, "close": 0, "area_pop": 0.0, "area_all": 0.0} for k in REPORT}
           for g in geoms}
    by_class = {g: {} for g in geoms}
    n_windows = 0
    t0 = time.time()
    for r in recs:
        rec = load_recording(a.data_dir, r, step=d.step)
        tm = rec.tracks_meta
        tr = rec.tracks.join(tm[["length", "width", "class"]], on="trackId")
        frames = {int(f): g for f, g in tr.groupby("frame", sort=False)}
        corners_by_frame = {f: car_corners(g["x"], g["y"], g["heading"], g["length"], g["width"])
                            for f, g in frames.items()}
        by_track = {t: g for t, g in rec.tracks.groupby("trackId")}
        meta_area = (tm["length"] * tm["width"]).to_dict()
        meta_cls = tm["class"].to_dict()
        n_rec = 0
        for t in eligible_ego_tracks(rec, cfg):
            g = by_track[t]
            if len(g) < L:
                continue
            fr, xs, ys, hs = (g[c].to_numpy() for c in ("frame", "x", "y", "heading"))
            # visible sets of every frame of this ego track, per geometry (computed once, reused by windows)
            vis = {geo: [] for geo in geoms}
            dist = []
            for j in range(len(g)):
                o = frames[int(fr[j])]
                cor = corners_by_frame[int(fr[j])]
                not_ego = (o["trackId"].to_numpy() != t)
                ids = o["trackId"].to_numpy()
                dist.append(dict(zip(ids[not_ego], np.hypot(o["x"].to_numpy() - xs[j],
                                                            o["y"].to_numpy() - ys[j])[not_ego])))
                for geo, (ah, be, hw) in geoms.items():
                    m = visible_mask(cor, xs[j], ys[j], hs[j], ah, be, hw) & not_ego
                    vis[geo].append(set(ids[m].tolist()))
            for s in range(0, len(g) - L + 1, stride):
                n_windows += 1; n_rec += 1
                for geo in geoms:
                    seen = set().union(*vis[geo][s:s + n])
                    closest = {}
                    for k in range(1, d.n_pred + 1):
                        j = s + n - 1 + k
                        new = vis[geo][j] - seen
                        for v in new:                                   # closest approach so far, per pop-in
                            closest[v] = min(closest.get(v, np.inf), dist[j].get(v, np.inf))
                        if k in REPORT:
                            A = acc[geo][k]
                            A["win"] += 1
                            A["popin"] += bool(new)
                            A["close"] += any(c <= a.close_m for c in closest.values())
                            A["area_pop"] += sum(meta_area[v] for v in new)
                            A["area_all"] += sum(meta_area[v] for v in vis[geo][j])
                            if k == REPORT[-1]:
                                for v in new:
                                    by_class[geo][meta_cls[v]] = by_class[geo].get(meta_cls[v], 0) + 1
        print(f"  rec {r:02d}: {n_rec} windows ({time.time() - t0:.0f} s)")

    rows = []
    print(f"\n{n_windows} windows from recordings {recs}")
    print(f"{'geometry':10s} {'step':>5s} {'windows_with_popin':>19s} {'popin_area_share':>17s} "
          f"{'close_popin(<' + str(int(a.close_m)) + 'm)':>18s}")
    for geo in geoms:
        for k in REPORT:
            A = acc[geo][k]
            row = {"geometry": geo, "step": k, "time_s": k * d.dt, "windows": A["win"],
                   "windows_with_popin": A["popin"] / A["win"], "popin_area_share": A["area_pop"] / max(A["area_all"], 1e-9),
                   "close_popin": A["close"] / A["win"]}
            rows.append(row)
            print(f"{geo:10s} {k * d.dt:4.0f}s {row['windows_with_popin']:19.3f} {row['popin_area_share']:17.3f} "
                  f"{row['close_popin']:18.3f}")
    print("\npop-ins at 4 s by class (count of road users):")
    for geo in geoms:
        print(f"  {geo:10s} " + ", ".join(f"{c} {n_}" for c, n_ in sorted(by_class[geo].items(), key=lambda x: -x[1])))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
