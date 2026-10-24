"""Turn one rounD recording into cached training arrays.

For every eligible ego track we render its whole life (at 5 Hz) once, and store:
    frames.npy   uint8   (N, 3, H, W)   ego-centric images, all ego tracks concatenated
    states.npy   float32 (N, 4)         ego x, y, heading (rad), speed
    actions.npy  float32 (N, 2)         action moving row i -> row i+1 (acceleration, curvature);
                                        the last row of each track is 0 and never used
    tracks.csv                          trackId, start row, length (rows)
    windows.npy  int64   (M,)           start row of every training window (never crosses a track)

Windows overlap, so frames are stored once per track rather than once per window.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .config import Config
from .kinematics import extract_actions
from .render import RoadMap, render_frame
from .round_io import Recording


def _frame_lookup(rec: Recording) -> dict[int, dict[str, np.ndarray]]:
    """frame -> arrays of every road user present in that frame (with its size and id)."""
    tm = rec.tracks_meta
    df = rec.tracks.join(tm[["length", "width"]], on="trackId")
    out = {}
    for frame, g in df.groupby("frame", sort=False):
        out[int(frame)] = {k: g[k].to_numpy() for k in ("trackId", "x", "y", "heading", "length", "width")}
    return out


def eligible_ego_tracks(rec: Recording, cfg: Config) -> list[int]:
    d = cfg.data
    counts = rec.tracks.groupby("trackId").size()
    cls = rec.tracks_meta["class"]
    return [int(t) for t, n in counts.items() if cls.get(t) in d.ego_classes and n >= d.cache_min_track_len]


def build_recording(rec: Recording, cfg: Config, out_dir: str, verbose: bool = True) -> dict:
    d, rc = cfg.data, cfg.render
    road = RoadMap(rec.background_path, rec.background_m_per_px(d.background_scale_down), rc)
    lookup = _frame_lookup(rec)
    by_track = {t: g for t, g in rec.tracks.groupby("trackId")}
    egos = eligible_ego_tracks(rec, cfg)

    n_total = sum(len(by_track[t]) for t in egos)
    os.makedirs(out_dir, exist_ok=True)
    frames = np.lib.format.open_memmap(os.path.join(out_dir, "frames.npy"), mode="w+",
                                       dtype=np.uint8, shape=(n_total, 3, rc.size, rc.size))
    states = np.zeros((n_total, 4), np.float32)
    actions = np.zeros((n_total, 2), np.float32)
    track_rows, windows = [], []

    row = 0
    for i, t in enumerate(egos):
        g = by_track[t]
        n = len(g)
        L, W = rec.tracks_meta.at[t, "length"], rec.tracks_meta.at[t, "width"]
        st = g[["x", "y", "heading", "speed"]].to_numpy()
        states[row:row + n] = st
        actions[row:row + n - 1] = extract_actions(st[:, 2], st[:, 3], d.dt, d.v_min_curvature,
                                                   d.accel_clip, d.curvature_clip)
        for j, (frame, x, y, h) in enumerate(zip(g["frame"], st[:, 0], st[:, 1], st[:, 2])):
            o = lookup[int(frame)]
            mask = o["trackId"] != t
            others = {k: o[k][mask] for k in ("x", "y", "heading", "length", "width")}
            ego = {"x": x, "y": y, "heading": h, "length": L, "width": W}
            frames[row + j] = render_frame(ego, others, road, rc)
        track_rows.append((t, row, n))
        windows.extend(range(row, row + n - d.cache_min_track_len + 1, d.window_stride))
        row += n
        if verbose and (i + 1) % 50 == 0:
            print(f"  rec {rec.rec_id}: {i + 1}/{len(egos)} ego tracks")

    frames.flush()
    np.save(os.path.join(out_dir, "states.npy"), states)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    np.save(os.path.join(out_dir, "windows.npy"), np.asarray(windows, np.int64))
    pd.DataFrame(track_rows, columns=["trackId", "start", "length"]).to_csv(
        os.path.join(out_dir, "tracks.csv"), index=False)
    info = {"rec_id": rec.rec_id, "location_id": rec.location_id, "n_ego_tracks": len(egos),
            "n_rows": int(n_total), "n_windows": len(windows)}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def split_recordings(rec_ids: list[int], cfg: Config) -> dict[str, list[int]]:
    """Deterministic split by whole recordings: the last ids go to test, the ones before to val."""
    ids = sorted(rec_ids)
    nt, nv = cfg.data.n_test_recordings, cfg.data.n_val_recordings
    if len(ids) <= nt + nv:
        raise ValueError(f"only {len(ids)} recordings; need more than {nt + nv}")
    return {"train": ids[:-(nt + nv)], "val": ids[-(nt + nv):-nt], "test": ids[-nt:]}
