"""Precompute the masks of the action-invariance loss (roundabout_wm/invariance.py): for every window of the cache,
which road users cannot react to the ego, as ego-centric footprints for the H steps after the warm-up. CPU only.

    python scripts/build_inv_masks.py --cache ../round_cache --data_dir ../round_data --out ../round_cache/inv_masks

Per recording (only the train split by default) it writes, in --out:
    rec_NN_starts.npy   int64 (M,)              window start rows (same grid as the dataset's windows)
    rec_NN_nr.npy       uint8 (M, H, 512)       footprints of road users that cannot react, bit-packed 64x64
    rec_NN_re.npy       uint8 (M, H, 512)       footprints of every other road user present
Step k (0 .. H-1) is the frame n_cond + k of the window, seen from the logged ego pose. About 10 KB per window
(~1.2 GB for the train split); read with mmap during training. Derived from rounD: keep it on HPC.
Classification: invariance.classify_others (lead-like on the ego's path, parked or receding away from it).
Prints the share of road users per class.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from roundabout_wm.build import _frame_lookup
from roundabout_wm.config import Config
from roundabout_wm.dataset import window_starts
from roundabout_wm.invariance import classify_others, path_polyline
from roundabout_wm.render import render_frame
from roundabout_wm.round_io import load_recording


def build_one(args):
    rec_id, cache, data_dir, out, H, lat, clear = args
    cfg = Config()
    d, rc = cfg.data, cfg.render
    n = d.n_cond
    rec = load_recording(data_dir, rec_id, step=d.step)
    look = _frame_lookup(rec)
    pos = {}                                                        # trackId -> (frames, xy, speed)
    for t, g in rec.tracks.groupby("trackId"):
        pos[int(t)] = (g["frame"].to_numpy(), g[["x", "y"]].to_numpy(), g["speed"].to_numpy())
    rdir = os.path.join(cache, f"rec_{rec_id:02d}")
    tcsv = pd.read_csv(os.path.join(rdir, "tracks.csv"))
    states = np.load(os.path.join(rdir, "states.npy"))
    starts = window_starts(rdir, n + H, d.window_stride)
    M = len(starts)
    nr = np.zeros((M, H, rc.size * rc.size // 8), np.uint8)
    re = np.zeros_like(nr)
    counts = {"lead": 0, "parked": 0, "receding": 0, "other": 0}
    meta = rec.tracks_meta
    for _, tr in tcsv.iterrows():
        tid, start, length = int(tr.trackId), int(tr.start), int(tr.length)
        frames = pos[tid][0]
        if len(frames) != length:
            raise RuntimeError(f"rec {rec_id} track {tid}: {len(frames)} raw rows vs {length} cached")
        L, W = float(meta.at[tid, "length"]), float(meta.at[tid, "width"])
        sel = np.flatnonzero((starts >= start) & (starts + n + H <= start + length))
        for m in sel:
            j0 = int(starts[m]) - start + n - 1                      # index of t0 (last warm-up frame) in the track
            r0 = start + j0
            ego_xy = states[r0:start + length, :2]
            path = path_polyline(ego_xy, states[r0, 2])
            past = path_polyline(states[start:r0 + 1, :2][::-1], states[r0, 2] + np.pi, length=40.0)
            f_win = frames[j0:j0 + H + 1]                           # t0 .. t0 + H
            o = look[int(f_win[0])]
            others0, futures = {}, {}
            for q, aid in enumerate(o["trackId"]):
                aid = int(aid)
                if aid == tid:
                    continue
                others0[aid] = (o["x"][q], o["y"][q], o["heading"][q])
                af, axy, av = pos[aid]
                k = np.searchsorted(af, f_win)
                ok = (k < len(af)) & (af[np.minimum(k, len(af) - 1)] == f_win)
                futures[aid] = (axy[k[ok]], av[k[ok]], states[r0 + np.flatnonzero(ok), :2])
            cls = classify_others(path, states[r0, :2], others0, futures, lat=lat, clear=clear, ego_len=L, past=past)
            for c in cls.values():
                counts[c] += 1
            nr_ids = np.array([a for a, c in cls.items() if c != "other"], dtype=np.int64)
            for k in range(H):
                ok_ = look[int(f_win[k + 1])]
                keep = ok_["trackId"] != tid
                isnr = np.isin(ok_["trackId"], nr_ids) & keep
                isre = ~np.isin(ok_["trackId"], nr_ids) & keep
                x, y, h = states[r0 + k + 1, :3]
                ego = {"x": x, "y": y, "heading": h, "length": L, "width": W}
                for dst, msk in ((nr, isnr), (re, isre)):
                    if msk.any():
                        sub = {kk: ok_[kk][msk] for kk in ("x", "y", "heading", "length", "width")}
                        fp = render_frame(ego, sub, None, rc)[1] > 25
                        dst[m, k] = np.packbits(fp.ravel())
    np.save(os.path.join(out, f"rec_{rec_id:02d}_starts.npy"), starts.astype(np.int64))
    np.save(os.path.join(out, f"rec_{rec_id:02d}_nr.npy"), nr)
    np.save(os.path.join(out, f"rec_{rec_id:02d}_re.npy"), re)
    return rec_id, M, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--H", type=int, default=10, help="steps after the warm-up (10 = 2 s)")
    p.add_argument("--lat", type=float, default=2.5, help="m from the ego's path that still counts as on it (lead)")
    p.add_argument("--clear", type=float, default=8.0, help="m a parked/receding road user stays away from the path")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    split = json.load(open(os.path.join(a.cache, "split.json")))
    recs = sorted({r for s in a.splits for r in split[s]})
    json.dump({"H": a.H, "lat": a.lat, "clear": a.clear, "recordings": recs},
              open(os.path.join(a.out, "info.json"), "w"), indent=1)
    t0, total, counts = time.time(), 0, {}
    jobs = [(r, a.cache, a.data_dir, a.out, a.H, a.lat, a.clear) for r in recs]
    with Pool(min(a.workers, len(jobs))) as pool:
        for rec_id, M, c in pool.imap_unordered(build_one, jobs):
            total += M
            counts = {k: counts.get(k, 0) + v for k, v in c.items()}
            print(f"  rec {rec_id}: {M} windows ({time.time() - t0:.0f} s)")
    tot = max(sum(counts.values()), 1)
    print(f"{total} windows; road users at t0: " + ", ".join(f"{k} {v / tot:.1%}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
