"""Where do seen cars get lost? A ladder of rungs, each adding one difficulty, scored car by car.

    python scripts/ladder.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/cnn_det/best.pt runs/ab_anchor/last.pt runs/ab_anchor_z1d/last.pt --n_windows 500

Rungs (all predict the SAME target frames n .. n+T-1, so columns are comparable):
    truth        the real frames scored against themselves: sanity check (kept ~1, split ~0, invented ~0)
    recon        decode h_j right after Pred has READ the real frame j (the warm-up path, run over the whole window):
                 encoder + 8x8 grid + decoder capacity, no prediction at all
    1step_post   from h_{j-1} (real frames up to j-1), one rollout step to frame j, z from the posterior
    1step_prior  the same with a prior sample: one-step prediction as the model would make it
    roll_post    20-step rollout from the warm-up, z from the posterior at every step: memory in h over many
                 steps, given the right latent
    roll_prior   20-step rollout, prior samples (what a controller gets); --k samples per window
    roll_pmean   20-step rollout with the prior MEAN (no sampling): cars kept here but not in roll_prior = the
                 sampling noise damages h; lost here too = the prior's mean itself is wrong
    mix_L<L>     20-step rollout, L prior samples then 1 posterior step, repeated (L = 1, 2, 5, 10):
                 cars kept at small L but lost at large L = the model never practises long runs of prior steps
                 (training's z-dropout gives runs of ~1-2); lost at every L = the prior is wrong more basically
A drop between two rungs points at the part that rung adds. Deterministic models: post = prior (z = 0).

Scoring per target frame (all geometry in the logged ego pose of that frame, from the raw rounD tracks):
    seen car     a road user visible in the crop during the warm-up; arrivals = the others visible later
    corridor     the car's true footprints from (t - tol) .. (t + tol), dilated by --dilate px: where a sample that
                 commits to a slightly different timing may legitimately put it. Pixels in several corridors go to
                 the car whose TRUE footprint is nearest. Green owned by no corridor is "invented".
    kept         predicted green in the car's region / its true area (pooled over car-steps); ~1 = kept, < 1 = fading
    faded        share of car-steps with kept < 0.5
    split        share of car-steps (with >= 1 blob) whose region holds >= 2 separate blobs (> --thr, >= --min_px):
                 one car drawn twice, e.g. at two timings
    lost         share of car-steps with no blob at all
    sharp        sum g^2 / sum g inside the seen cars' regions, relative to the truth's (1 = as solid as the truth)
    invented     green owned by no corridor / all true green
    arrivals     green in arrivals' regions / their true area
    lead         kept / split for the "lead" road user: seen, at the last warm-up frame 0-25 m ahead and < 2.5 m to
                 the side of the ego, the nearest such (the one a controller's cost depends on most)
Only car-steps with at least --min_area px of true footprint in view are scored.
Writes ladder.csv (ckpt, rung, step, all sums) next to the first checkpoint unless --out. CPU is fine.
Rendered rounD data: run on HPC / your own machine only.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pandas as pd
import torch

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.render import car_corners, render_frame

REPORT = (1, 5, 10, 20)
BASE_RUNGS = ("truth", "recon", "1step_post", "1step_prior", "roll_post", "roll_prior", "roll_pmean")
MIX_L = (1, 2, 5, 10)
RUNGS = BASE_RUNGS + tuple(f"mix_L{L}" for L in MIX_L)
SUMS = ("kept_num", "kept_den", "n_car", "n_faded", "n_blob1", "n_split", "n_lost", "sq", "s", "sq_t", "s_t",
        "invented", "true_all", "arr_num", "arr_den", "lead_num", "lead_den", "n_lead", "n_lead_split")


# ---- geometry (pure numpy; tested in tests/test_ladder.py) ----------------------------------------------------
def render_poses(poses, ego, rc):
    """poses: list of (x, y, heading, length, width) -> (64, 64) float in [0, 1], all drawn in the ego's frame."""
    if not poses:
        return np.zeros((rc.size, rc.size), np.float32)
    p = np.asarray(poses, np.float64)
    others = dict(x=p[:, 0], y=p[:, 1], heading=p[:, 2], length=p[:, 3], width=p[:, 4])
    e = dict(x=float(ego[0]), y=float(ego[1]), heading=float(ego[2]), length=4.5, width=1.8)
    return render_frame(e, others, None, rc)[1].astype(np.float32) / 255.0


def owner_map(footprints, corridors):
    """footprints, corridors: lists (one per car) of (H, W) float / bool. -> (H, W) int: the index of the car owning
    each pixel (-1: none). A pixel belongs to a car if it is in that car's corridor; ties go to the car whose true
    footprint is nearest. Cars with an empty footprint (out of view now) only win pixels no visible car claims."""
    if not footprints:
        return None
    H, W = corridors[0].shape
    d = np.full((len(footprints), H, W), np.inf, np.float32)
    for c, (f, cor) in enumerate(zip(footprints, corridors)):
        m = f > 0.1
        dist = (cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if m.any()
                else np.full((H, W), 1e3, np.float32))
        d[c][cor] = dist[cor]
    own = d.argmin(0)
    own[~np.isfinite(d.min(0))] = -1
    return own.astype(np.int16)


def count_blobs(mask, min_px):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_px).sum())


def score_step(pred, own, cars, t, acc, thr, min_px, min_area):
    """Add one target frame's sums to `acc` (dict of arrays indexed by step t).
    pred (H, W) predicted green; own: owner_map; cars: list of dicts with seen, lead, area (T,), sq_t (T,)."""
    acc["true_all"][t] += sum(c["area"][t] for c in cars)
    if own is None:
        acc["invented"][t] += pred.sum()
        return
    acc["invented"][t] += pred[own < 0].sum()
    for ci, c in enumerate(cars):
        reg = own == ci
        g = pred[reg]
        if not c["seen"]:
            acc["arr_num"][t] += g.sum(); acc["arr_den"][t] += c["area"][t]
            continue
        if c["area"][t] < min_area:
            continue
        kept = g.sum() / c["area"][t]
        nb = count_blobs((pred > thr) & reg, min_px)
        acc["kept_num"][t] += g.sum(); acc["kept_den"][t] += c["area"][t]; acc["n_car"][t] += 1
        acc["n_faded"][t] += kept < 0.5
        acc["n_blob1"][t] += nb >= 1; acc["n_split"][t] += nb >= 2; acc["n_lost"][t] += nb == 0
        acc["sq"][t] += (g * g).sum(); acc["s"][t] += g.sum()
        acc["sq_t"][t] += c["sq_t"][t]; acc["s_t"][t] += c["area"][t]
        if c["lead"]:
            acc["lead_num"][t] += g.sum(); acc["lead_den"][t] += c["area"][t]; acc["n_lead"][t] += 1
            acc["n_lead_split"][t] += nb >= 2


def window_geometry(win, st, cfg, meta, tol, dilate, lead_ahead=25.0, lead_side=2.5):
    """win: list of n+T DataFrames (road users per frame, indexed by trackId, ego dropped); st (n+T, 4) ego x, y,
    heading, speed. -> (owners: list of T owner maps, cars: list of dicts)."""
    from fov_popin import crops, visible_mask
    n, T, rc = cfg.data.n_cond, cfg.data.n_pred, cfg.render
    L = n + T
    box = crops(cfg)["current"]

    def pose(j, tid):
        r = win[j].loc[tid]
        return (r["x"], r["y"], r["heading"], r["length"], r["width"])

    def visible(j, tid):
        r = win[j].loc[tid]
        cor = car_corners(r["x"], r["y"], r["heading"], r["length"], r["width"])
        return bool(visible_mask(cor, st[j, 0], st[j, 1], st[j, 2], *box)[0])

    seen = {tid for j in range(n) for tid in win[j].index if visible(j, tid)}
    later = {tid for j in range(n, L) for tid in win[j].index if visible(j, tid)} - seen
    # lead: seen, present at the last warm-up frame, ahead of the ego and nearly in line with it
    lead, best = None, np.inf
    c, s = np.cos(st[n - 1, 2]), np.sin(st[n - 1, 2])
    for tid in seen:
        if tid not in win[n - 1].index:
            continue
        dx, dy = win[n - 1].at[tid, "x"] - st[n - 1, 0], win[n - 1].at[tid, "y"] - st[n - 1, 1]
        fwd, left = dx * c + dy * s, -dx * s + dy * c
        if 0 < fwd <= lead_ahead and abs(left) < lead_side and fwd < best:
            lead, best = tid, fwd
    ids = sorted(seen) + sorted(later)
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    cars = [dict(id=int(tid), seen=tid in seen, lead=tid == lead, area=np.zeros(T), sq_t=np.zeros(T),
                 cls=str(meta.loc[tid, "class"]) if meta is not None and "class" in meta.columns and tid in meta.index
                 else "?") for tid in ids]
    owners = []
    for t in range(T):
        j = n + t
        fps, cors = [], []
        for ci, tid in enumerate(ids):
            fp = render_poses([pose(j, tid)] if tid in win[j].index else [], st[j], rc)
            near = [pose(k, tid) for k in range(max(0, j - tol), min(L, j + tol + 1)) if tid in win[k].index]
            cor = cv2.dilate((render_poses(near, st[j], rc) > 0.1).astype(np.uint8), kernel) > 0
            fps.append(fp); cors.append(cor)
            cars[ci]["area"][t] = fp.sum(); cars[ci]["sq_t"][t] = (fp * fp).sum()
        owners.append(owner_map(fps, cors))
    return owners, cars


# ---- model rungs -------------------------------------------------------------------------------------------------
@torch.no_grad()
def scheduled_rollout(m, b, n, T, source):
    """Rollout like WorldModel.forward, but z at step k comes from source(k) in {"post", "prior", "pmean"}.
    Same random-number use as forward() for all-"post" / all-"prior" schedules (tested). -> (B, T, 64, 64)."""
    imgs, act, spd = b["images"], b["actions"], b["speed"]
    B = imgs.shape[0]
    e = m.encode(imgs[:, :n + T])
    h, _ = m.warmup(e[:, :n], act, spd)
    prev = imgs[:, n - 1, GREEN:GREEN + 1]
    out = []
    for k in range(T):
        i = n - 1 + k
        frame = None
        if m.cfg.reencode:
            frame = imgs[:, i] if k == 0 else torch.cat([imgs[:, i, :GREEN], prev.to(imgs.dtype),
                                                         imgs[:, i, GREEN + 1:]], 1)
        if m.cfg.deterministic:
            z = m.zero_z(B, h.device)
        else:
            src = source(k)
            mu_p, lv_p = m.prior_params(h)
            if src == "post":
                z = m._sample(*m.posterior(torch.cat([h, e[:, i + 1]], 1)))
            elif src == "prior":
                z = m._sample(mu_p, lv_p)
            else:
                z = mu_p
        h, g = m.rollout_step(h, z, act[:, i], spd[:, i], frame, prev)
        out.append(g); prev = g
    return torch.stack(out, 1)[:, :, 0].float()


@torch.no_grad()
def rungs(m, b, n, T, k, mix_L=MIX_L):
    """-> dict rung -> list of (B, T, 64, 64) predictions (one per sample)."""
    imgs, act, spd = b["images"], b["actions"], b["speed"]
    B, det = imgs.shape[0], m.cfg.deterministic
    real_g = imgs[:, :, GREEN]
    e = m.encode(imgs[:, :n + T])
    _, hs = m.warmup(e, act, spd)                                  # Pred reads every real frame 0 .. n+T-1
    out = {"truth": [real_g[:, n:n + T]]}
    out["recon"] = [torch.stack([m.decode(hs[j], real_g[:, j - 1:j]) for j in range(n, n + T)], 1)[:, :, 0]]
    H = torch.stack(hs[n - 1:n + T - 1], 1).flatten(0, 1)          # h_{j-1} for every target j
    a1, s1 = act[:, n - 1:n + T - 1].flatten(0, 1), spd[:, n - 1:n + T - 1].flatten(0, 1)
    fr = imgs[:, n - 1:n + T - 1].flatten(0, 1) if m.cfg.reencode else None
    prev = imgs[:, n - 1:n + T - 1, GREEN:GREEN + 1].flatten(0, 1)

    def one(source):
        if det:
            z = m.zero_z(B * T, imgs.device)
        elif source == "post":
            z = m._sample(*m.posterior(torch.cat([H, e[:, n:n + T].flatten(0, 1)], 1)))
        else:
            z = m._sample(*m.prior_params(H))
        return m.rollout_step(H, z, a1, s1, fr, prev)[1].float().view(B, T, 64, 64)
    out["1step_post"] = [one("post")]
    out["1step_prior"] = [one("prior")]
    out["roll_post"] = [m(imgs, act, spd, T, z_source="post")["green"][:, :, 0].float()]
    out["roll_prior"] = [m(imgs, act, spd, T, z_source="prior")["green"][:, :, 0].float() for _ in range(k)]
    out["roll_pmean"] = [scheduled_rollout(m, b, n, T, lambda s: "pmean")]
    for L in mix_L:                                   # L prior steps, then one posterior step, repeated
        out[f"mix_L{L}"] = [scheduled_rollout(m, b, n, T, lambda s, L=L: "post" if s % (L + 1) == L else "prior")
                            for _ in range(k)]
    return out


def summarize(acc, t):
    d = lambda a, b: a / b if b > 0 else float("nan")
    return dict(kept=d(acc["kept_num"][t], acc["kept_den"][t]), faded=d(acc["n_faded"][t], acc["n_car"][t]),
                split=d(acc["n_split"][t], acc["n_blob1"][t]), lost=d(acc["n_lost"][t], acc["n_car"][t]),
                sharp=d(d(acc["sq"][t], acc["s"][t]), d(acc["sq_t"][t], acc["s_t"][t])),
                invented=d(acc["invented"][t], acc["true_all"][t]), arrivals=d(acc["arr_num"][t], acc["arr_den"][t]),
                lead_kept=d(acc["lead_num"][t], acc["lead_den"][t]),
                lead_split=d(acc["n_lead_split"][t], acc["n_lead"][t]))


def main():
    from leakage import WindowTracks
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=500)
    p.add_argument("--k", type=int, default=1, help="prior samples per window for roll_prior")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=int, default=5, help="corridor timing tolerance in steps (5 = +-1 s)")
    p.add_argument("--dilate", type=int, default=2, help="corridor dilation in px (2 = 1.5 m)")
    p.add_argument("--thr", type=float, default=0.3, help="intensity above which a pixel counts for blobs")
    p.add_argument("--min_px", type=int, default=3, help="smallest blob counted")
    p.add_argument("--min_area", type=float, default=4.0, help="px of true footprint for a car-step to be scored")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]

    # geometry once, shared by all checkpoints (stats only change the normalised actions, not the frames)
    s0 = torch.load(a.ckpts[0], map_location="cpu")
    ds0 = RounDWindows(a.cache, split, cfg, s0["stats"], n_pred=T)
    idx = np.sort(np.random.default_rng(a.seed).choice(len(ds0), min(a.n_windows, len(ds0)), replace=False))
    tracks = WindowTracks(SimpleNamespace(cfg=cfg, ds=ds0, rec_ids=split), a.cache, a.data_dir)
    geo, n_lead, classes = {}, 0, {}
    for c, i in enumerate(idx):
        win, st = tracks.window(int(i))
        k_rec, _ = ds0.index[int(i)]
        meta = tracks._rec[split[int(k_rec)]][3]
        geo[int(i)] = window_geometry(win, st, cfg, meta, a.tol, a.dilate)
        for car in geo[int(i)][1]:
            if car["lead"]:
                n_lead += 1; classes[car["cls"]] = classes.get(car["cls"], 0) + 1
        if (c + 1) % 100 == 0:
            print(f"  geometry {c + 1}/{len(idx)} windows")
    print(f"{len(idx)} {a.split} windows | {n_lead} with a lead road user ({classes}) | "
          f"corridor +-{a.tol * dt:.1f} s, dilate {a.dilate} px")

    rows = []
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=T)
        acc = {r: {key: np.zeros(T) for key in SUMS} for r in RUNGS}
        for j in range(0, len(idx), a.batch):
            items = [ds[int(i)] for i in idx[j:j + a.batch]]
            b = {key: torch.stack([it[key] for it in items]).to(dev) for key in items[0]}
            torch.manual_seed(a.seed + j)
            out = rungs(m, b, n, T, a.k)
            for r, preds in out.items():
                for pred in preds:
                    pred = pred.cpu().numpy()
                    for bi, i in enumerate(idx[j:j + a.batch]):
                        owners, cars = geo[int(i)]
                        for t in range(T):
                            score_step(pred[bi, t], owners[t], cars, t, acc[r], a.thr, a.min_px, a.min_area)
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}/{os.path.basename(path)}"
        print(f"\n{name} ({'deterministic' if m.cfg.deterministic else 'stochastic'}, step {s.get('step', '?')})")
        print(f"{'rung':12s} {'time':>5s} {'kept':>6s} {'faded':>6s} {'split':>6s} {'lost':>6s} {'sharp':>6s} "
              f"{'invent':>7s} {'arriv':>6s} {'lead':>6s} {'l.split':>7s}")
        for r in RUNGS:
            for t in range(T):
                sm = summarize(acc[r], t)
                rows.append(dict(ckpt=name, rung=r, time_s=round((t + 1) * dt, 2), **sm,
                                 **{f"sum_{key}": acc[r][key][t] for key in SUMS}))
                if t + 1 in REPORT:
                    print(f"{r:12s} {(t + 1) * dt:4.1f}s {sm['kept']:6.2f} {sm['faded']:6.2f} {sm['split']:6.2f} "
                          f"{sm['lost']:6.2f} {sm['sharp']:6.2f} {sm['invented']:7.2f} {sm['arrivals']:6.2f} "
                          f"{sm['lead_kept']:6.2f} {sm['lead_split']:7.2f}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpts[0])), "ladder.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
