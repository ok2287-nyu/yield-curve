"""Leakage test: does the ego's plan change the predicted traffic where the ego CANNOT cause anything?

    python scripts/leakage.py --ckpt runs/cnn_det/best.pt --cache ../round_cache --data_dir ../round_data \
        --n_scenes 300 --pairs brake:go,brake:coast,coast:go

Why: the net-mass score of reaction_map.py cannot tell leakage (the plan used as evidence: "people go when the ring is
clear") from a real reaction (a car up the ring yields and enters the view later). This test only looks where no
real reaction is possible.

Receding road users: visible in the crop during the warm-up, present for the whole 4 s, and, for BOTH plans,
    - their (logged) distance to the ego never shrinks by more than --tol m per step, and
    - their logged path never comes within --clear m of the ego's path at ANY pair of times (so they are not a car
      ahead in the ego's lane, a follower, or a car crossing the ego's route).
The ego cannot delay, speed up, create or delete such a road user within 4 s. (Their logged motion is what the human
ego's plan produced, but by construction it does not depend on the ego.)

Per step, in plan A's ego frame, regions around the road users' TRUE positions (footprint dilated by --dilate px),
inside the area both plans see (as in reaction_map.py):
    receding     around receding road users, minus anything near another road user
    interacting  around the other road users seen during the warm-up (followers, yielders, crossers ...)
Scores (pooled over scenes: sum of |B - A| in the region / sum of the true footprint area in the region):
    receding     LEAKAGE: should be at the floor
    interacting  the real reaction plus any leakage, for scale
    floor        the same measure for a model with NO leakage: plan A's prediction warped to plan B's frame and back
                 (pure resampling error; an upper bound, as it resamples twice)
    net          signed version of receding (B - A): negative = the faster plan deletes receding traffic
    coverage     plan A's predicted mass inside the receding region / true footprint area: if low, the model's cars
                 are not where the region expects them and the receding score means little
Writes <ckpt dir>/leakage.csv (per scene, pair, step: the sums) and prints the scores every 0.4 s.
Rendered rounD data: run on HPC / your own machine only.
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

from fov_popin import crops, visible_mask
from reaction_map import compare
from roundabout_wm.render import car_corners, render_frame
from roundabout_wm.round_io import load_recording
from roundabout_wm.sim import ImaginedDrive
from roundabout_wm.wm_eval import warp_previous


class WindowTracks:
    """The raw road users of a dataset window: per window frame, a DataFrame indexed by trackId."""

    def __init__(self, sim, cache, data_dir):
        self.sim, self.cache, self.data_dir = sim, cache, data_dir
        self._rec = {}

    def _load(self, rec_id):
        if rec_id not in self._rec:
            d = self.sim.cfg.data
            rec = load_recording(self.data_dir, rec_id, step=d.step)
            tr = rec.tracks.join(rec.tracks_meta[["length", "width"]], on="trackId")
            frames = {int(f): g.set_index("trackId") for f, g in tr.groupby("frame", sort=False)}
            ego_frames = {int(t): g["frame"].to_numpy() for t, g in rec.tracks.groupby("trackId")}
            tracks_csv = pd.read_csv(os.path.join(self.cache, f"rec_{rec_id:02d}", "tracks.csv"))
            self._rec[rec_id] = (frames, ego_frames, tracks_csv, rec.tracks_meta)
        return self._rec[rec_id]

    def window(self, i):
        d = self.sim.cfg.data
        n, T = d.n_cond, d.n_pred
        k_rec, s_row = (int(x) for x in self.sim.ds.index[i])
        rec_id = self.sim.rec_ids[k_rec]
        frames, ego_frames, tracks_csv, meta = self._load(rec_id)
        row = tracks_csv[(tracks_csv.start <= s_row) & (s_row < tracks_csv.start + tracks_csv.length)].iloc[0]
        ego_id, off = int(row.trackId), s_row - int(row.start)
        fr = ego_frames[ego_id][off:off + n + T]
        st = self.sim.ds[i]["ego_state"].numpy()
        e = frames[int(fr[n - 1])].loc[ego_id]
        if not np.allclose([e["x"], e["y"]], st[n - 1, :2], atol=1e-3):
            raise RuntimeError(f"window {i}: cached ego pose does not match the raw track")
        return [frames[int(f)].drop(index=ego_id, errors="ignore") for f in fr], st


def classify(win, st, pA, pB, cfg, tol, clear):
    """-> (seen ids, receding ids). pA, pB: (T, 3) ego poses of the two plans."""
    n, T = cfg.data.n_cond, cfg.data.n_pred
    ahead, behind, half = crops(cfg)["current"]
    seen = set()
    for j in range(n):
        g = win[j]
        cor = car_corners(g["x"], g["y"], g["heading"], g["length"], g["width"])
        seen.update(g.index[visible_mask(cor, st[j, 0], st[j, 1], st[j, 2], ahead, behind, half)].tolist())
    e0 = st[n - 1, :2]
    receding = []
    for tid in seen:
        if not all(tid in win[n + t].index for t in range(T)) or tid not in win[n - 1].index:
            continue
        c = np.array([[win[n - 1 + t].at[tid, "x"], win[n - 1 + t].at[tid, "y"]] for t in range(T + 1)])  # (T+1, 2)
        ok = True
        for p in (pA, pB):
            e = np.vstack([e0, p[:, :2].numpy()])                                                      # (T+1, 2)
            dist = np.linalg.norm(c - e, axis=1)
            paths = np.linalg.norm(c[:, None] - e[None], axis=2)                                       # any times
            if (np.diff(dist) < -tol).any() or paths.min() < clear:
                ok = False
                break
        if ok:
            receding.append(tid)
    return seen, receding


def footprints(win, ids, pA, cfg, n):
    """(T, 64, 64) soft footprints of road users `ids` at their true positions, in plan A's ego frame."""
    T, rc = cfg.data.n_pred, cfg.render
    out = np.zeros((T, rc.size, rc.size), np.float32)
    for t in range(T):
        g = win[n + t]
        g = g.loc[[i for i in ids if i in g.index]]
        ego = dict(x=float(pA[t, 0]), y=float(pA[t, 1]), heading=float(pA[t, 2]), length=4.5, width=1.8)
        others = dict(x=g["x"].to_numpy(), y=g["y"].to_numpy(), heading=g["heading"].to_numpy(),
                      length=g["length"].to_numpy(), width=g["width"].to_numpy())
        out[t] = render_frame(ego, others, None, rc)[1] / 255.0
    return torch.from_numpy(out)


def dilate(soft, r):
    return F.max_pool2d((soft > 0.1).float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_scenes", type=int, default=300)
    p.add_argument("--pairs", default="brake:go,brake:coast,coast:go")
    p.add_argument("--k", type=int, default=8, help="prior samples averaged per plan (stochastic models)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=int, default=4, help="as in reaction_map.py")
    p.add_argument("--dilate", type=int, default=4, help="px around each true footprint (4 = 3 m)")
    p.add_argument("--tol", type=float, default=0.2, help="m a receding road user may come closer per step")
    p.add_argument("--clear", type=float, default=8.0, help="m between its path and the ego's path, at any times")
    p.add_argument("--min_area", type=float, default=200.0,
                   help="px of receding footprint (pooled) below which a step is not scored (~15 cars' worth = 200)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sim = ImaginedDrive.load(a.ckpt, a.cache, a.data_dir, a.split)
    cfg = sim.cfg
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    k = 1 if sim.model.cfg.deterministic else a.k
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
    tracks = WindowTracks(sim, a.cache, a.data_dir)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(sim.ds))

    rows, n_used, n_with = [], 0, 0
    for i in order:
        if n_used == a.n_scenes:
            break
        i = int(i)
        it = sim.ds[i]
        if it["images"][n - 1, 1].sum() < it["images"][n - 1, 2].sum():        # no traffic: skip, as reaction_map
            continue
        n_used += 1
        win, st = tracks.window(i)
        any_rec = False
        for pa, pb in pairs:
            c = compare(sim, i, pa, pb, T, k, a.seed, a.margin)
            # poses again (compare keeps them internal): the plans are deterministic in the ego's motion
            pA, pB = c["pA"], c["pB"]
            seen, rec = classify(win, st, pA, pB, cfg, a.tol, a.clear)
            if not rec:
                continue
            any_rec = True
            others_all = {t for f in win[n:] for t in f.index}
            fp_rec = footprints(win, rec, pA, cfg, n)
            fp_int = footprints(win, [t for t in seen if t not in rec], pA, cfg, n)
            fp_rest = footprints(win, sorted(set(others_all) - set(rec)), pA, cfg, n)
            core = c["core"]
            m_rec = dilate(fp_rec, a.dilate) & ~dilate(fp_rest, a.dilate) & core
            m_int = dilate(fp_int, a.dilate) & ~dilate(fp_rec, a.dilate) & core
            # floor: A's prediction to B's frame and back (a model whose B prediction is A's, i.e. no leakage)
            back = warp_previous(warp_previous(c["gA"][:, None], pA, pB[:, None], cfg.render)[:, 0], pB,
                                 pA[:, None], cfg.render)[:, 0, 0]
            d, fl = c["diff"], back - c["gA"]
            for t in range(T):
                mr, mi = m_rec[t], m_int[t]
                rows.append(dict(
                    scene=i, a=pa, b=pb, time_s=round((t + 1) * dt, 2), n_receding=len(rec),
                    rec_abs=d[t][mr].abs().sum().item(), rec_net=d[t][mr].sum().item(),
                    rec_floor=fl[t][mr].abs().sum().item(), rec_area=fp_rec[t][mr].sum().item(),
                    rec_pred=c["gA"][t][mr].sum().item(),
                    int_abs=d[t][mi].abs().sum().item(), int_area=fp_int[t][mi].sum().item()))
        n_with += any_rec
        if n_used % 25 == 0:
            print(f"  {n_used}/{a.n_scenes} scenes, {n_with} with receding road users")

    out = a.out or os.path.join(os.path.dirname(a.ckpt), "leakage.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    kind = "deterministic" if k == 1 else f"mean of {k} prior samples"
    print(f"\n{n_used} scenes with traffic, {n_with} with receding road users | {kind} | "
          f"tol {a.tol} m/step, clear {a.clear} m, dilate {a.dilate} px")
    if df.empty:
        print("no receding road users found"); return
    for (pa, pb), g in df.groupby(["a", "b"], sort=False):
        print(f"\n{pa} vs {pb}: |B - A| per unit of true footprint area ({g.scene.nunique()} scenes)")
        print(f"{'time':>6s} {'receding':>9s} {'floor':>7s} {'net':>7s} {'coverage':>9s} {'interacting':>12s} "
              f"{'receding area (px)':>19s}")
        for t, h in g.groupby("time_s"):
            if round(t / dt) % 2:
                continue
            ra, ia = h.rec_area.sum(), max(h.int_area.sum(), 1e-6)
            if ra < a.min_area:                      # too little receding traffic in view to say anything
                print(f"{t:5.1f}s {'-':>9s} {'-':>7s} {'-':>7s} {'-':>9s} {h.int_abs.sum() / ia:12.3f} {ra:19.0f}")
                continue
            print(f"{t:5.1f}s {h.rec_abs.sum() / ra:9.3f} {h.rec_floor.sum() / ra:7.3f} {h.rec_net.sum() / ra:7.3f} "
                  f"{h.rec_pred.sum() / ra:9.2f} {h.int_abs.sum() / ia:12.3f} {ra:19.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
