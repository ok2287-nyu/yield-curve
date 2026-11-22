"""Does the learned prior know when a STOPPED road user is about to go?

    python scripts/yield_starts.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_anchor/last.pt runs/ab_inv4_01/last.pt --n_windows 3000

The uncertainty maps say the model is most wrong about slow / stopped road users at the entries, and that sigma_p
ranks those places right but widens only ~5 %. This is the direct test of the research question there.

1. Stopped road users, from the true frames only: pixels that are green (> 0.5) in the last observed frame n-1 and
   Connected components of green > 0.5 in the last observed frame n-1 (>= --min_area px, away from the crop
   border) whose green centroid moved < 0.5 px (0.375 m) since frames n-2 and n-3, once the ego's own motion is
   warped out: "stopped road users" (a road user creeping under ~0.9 m/s also counts).
2. Each one's footprint (its pixels at n-1, fixed in the WORLD) is warped into every future frame n+k with the logged
   ego poses. It has "started" at the first k where > 30 % of the footprint is no longer green (a car has moved
   ~1.4 m: from a standstill that lags the true start by a few tenths of a second). Labels:
       start_2s  starts within --horizon steps (default 10 = 2 s)
       start_4s  starts between 2 s and 4 s
       stays     still there after all --n_future steps (default 20 = 4 s)   (includes parked road users)
       gone      footprint left the crop before it could be decided (dropped)
3. sigma_p over the footprint, from the posterior-driven track (uncertainty_map.posterior_track): sigma_p at step k
   is the prior for frame n+k given the true frames up to n+k-1. For a road user that starts at k*, the steps before
   k* still show it standing, so nothing leaks.

Prints, per checkpoint:
    - counts per label, and where they are (world bins of 6 m with the most starters)
    - AUROC of sigma_p at step 1 (the first prediction, before any future frame) for start_2s vs stays, and for
      start_2s vs start_4s + stays. 0.5 = sigma_p says nothing about who goes; 1 = it ranks every starter above
      every stayer.
    - the same AUROC for the KL at step 1 (needs the future frame: an upper bound on what a prior could know)
    - mean sigma_p aligned to the start step (k - k* = -4 .. 0) for starters vs the stayers' mean at the same steps:
      does the prior widen as the start approaches?
    - with --region X Y R, everything again for road users whose footprint centre is within R m of (X, Y)
    - with --k K, K imagined futures (prior rollouts over --horizon) per window: in how many does the stopped road
      user drive off (footprint > 30 % empty while the green within 4.5 m stays >= 60 %) or vanish (green gone)?
      Per true label, the AUROC of that share for start_2s vs stays, and the imagined vs true start step. A narrow
      prior is only a failure if the model also does not IMAGINE the start: this says which.
Writes yield_starts_<run>.csv (one row per stopped road user) and yield_starts.png (sigma_p vs steps to start).
No rounD images in the outputs. CPU is fine but slow; a GPU job is faster.
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

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.wm_eval import warp_previous
from uncertainty_map import posterior_track


def _shift(a, b, reg):
    """Distance in px between the green centroids of a and b inside region reg (inf if b is ~empty there)."""
    ys, xs = torch.nonzero(reg, as_tuple=True)
    wa, wb = a[ys, xs].float(), b[ys, xs].float()
    if wb.sum() < 0.5 * wa.sum() or wa.sum() < 1e-6:
        return float("inf")
    ca = torch.stack([(wa * ys).sum(), (wa * xs).sum()]) / wa.sum()
    cb = torch.stack([(wb * ys).sum(), (wb * xs).sum()]) / wb.sum()
    return float((ca - cb).norm())


def _warp(x, pose_src, pose_tgt, rc):
    """x (64, 64) at pose_src (3,) -> (T, 64, 64) at each of pose_tgt (T, 3)."""
    return warp_previous(x[None, None].float(), pose_src[None], pose_tgt[None], rc)[0, :, 0]


@torch.no_grad()
def stopped_users(green, pose, n, n_future, rc, min_area=6, border=4, still=0.5, moved=0.3):
    """One window. green (T, 64, 64) true green in [0, 1], pose (T, 3) ego x, y, heading. ->
    list of dicts: mask (64, 64) bool at frame n-1, start (first k in 0..n_future-1 where the road user has moved,
    -1 = stayed, None = left the crop first), fp (n_future, 64, 64) bool footprint warped into frames n..n+n_future-1."""
    g1 = green[n - 1]
    back = _warp(green[n - 3], pose[n - 3], pose[n - 2:n], rc)                     # n-3 seen at n-2 and n-1
    prev = _warp(green[n - 2], pose[n - 2], pose[n - 1:n], rc)[0]
    num, lab, st, _ = cv2.connectedComponentsWithStats((g1 > 0.5).cpu().numpy().astype(np.uint8), connectivity=8)
    out = []
    for c in range(1, num):
        x0, y0, w, h, area = st[c]
        if area < min_area or x0 < border or y0 < border or x0 + w > 64 - border or y0 + h > 64 - border:
            continue
        cm = torch.from_numpy(lab == c).to(green.device)
        # stopped = the road user's centroid has not moved over the last 0.4 s once the ego's motion is warped out
        # (centroids, not pixels: a sharp re-render and a bilinear-warped copy differ at the edges even when parked)
        reg = torch.from_numpy(cv2.dilate((lab == c).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).to(green.device)
        if _shift(g1, back[1], reg) > still or _shift(g1, prev, reg) > still:
            continue
        mask = cm
        fp = _warp(mask, pose[n - 1], pose[n:n + n_future], rc) > 0.5                 # (n_future, 64, 64)
        near = torch.from_numpy(cv2.dilate((lab == c).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
        near = _warp(near.to(green.device), pose[n - 1], pose[n:n + n_future], rc) > 0.5  # +-4.5 m around it
        start = -1
        for k in range(n_future):
            if fp[k].sum() < 0.5 * area:                                               # footprint left the crop
                start = None
                break
            if (green[n + k][fp[k]] < 0.5).float().mean() > moved:
                start = k
                break
        out.append({"mask": mask, "start": start, "fp": fp, "near": near, "area": int(area),
                    "mass0": float(g1[torch.from_numpy(cv2.dilate((lab == c).astype(np.uint8),
                                                                  np.ones((13, 13), np.uint8)) > 0).to(green.device)].sum())})
    return out


def sample_start(g, u, moved=0.3, keep=0.6):
    """One imagined future g (H, 64, 64) for stopped road user u -> (k, faded): k = first step where > moved of its
    footprint is empty (-1 = never within H), faded = at that step the green within +-4.5 m of it fell below keep x
    what was there at n-1 (the road user vanished rather than drove off)."""
    for k in range(g.shape[0]):
        f = u["fp"][k]
        if not f.any():
            return -1, False
        if (g[k][f] < 0.5).float().mean() > moved:
            return k, float(g[k][u["near"][k]].sum()) < keep * u["mass0"]
    return -1, False


def auroc(pos, neg):
    """P(a random positive scores above a random negative), ties count half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(d, H, title):
    s2, s4, st = d[d.label == "start_2s"], d[d.label == "start_4s"], d[d.label == "stays"]
    print(f"  {title}: {len(d)} stopped road users: start within {H * 0.2:.0f} s {len(s2)}, "
          f"start later {len(s4)}, stay {len(st)}")
    row = {}
    for key, name in (("sig_0", "sigma_p"), ("kl_0", "KL")):
        if key not in d or not np.isfinite(d[key]).any():
            continue
        a1, a2 = auroc(s2[key], st[key]), auroc(s2[key], pd.concat([s4[key], st[key]]))
        print(f"    {name:7s} at step 1: starters {s2[key].mean():.3f} vs stayers {st[key].mean():.3f} | "
              f"AUROC start_2s vs stays {a1:.2f}, vs everyone else {a2:.2f}")
        row.update({f"auroc_{name}_vs_stays": a1, f"auroc_{name}_vs_rest": a2})
    if "p_start" in d:
        print(f"    imagined futures ({int(d.k_samples.iloc[0])} prior samples each): share of samples where it "
              f"drives off within {H * 0.2:.0f} s / where it vanishes instead")
        for lab, g in (("start within 2 s", s2), ("start 2-4 s", s4), ("stays", st)):
            if len(g):
                print(f"      truly {lab:17s} drives off {g.p_start.mean():.2f}   vanishes {g.p_fade.mean():.2f}   "
                      f"(n {len(g)})")
        a = auroc(s2.p_start, st.p_start)
        print(f"    AUROC of 'share of samples that drive off', start_2s vs stays: {a:.2f}")
        hit = s2[s2.k_pred >= 0]
        if len(hit):
            print(f"    timing where the samples drive off: imagined step {hit.k_pred.mean():.1f} vs true "
                  f"{hit.start.mean():.1f} (+1 = 0.2 s later), n {len(hit)}")
        row["auroc_samples"] = a
    if len(s2):
        cols = [f"sig_{k}" for k in range(H)]
        rel = {}
        for _, r in s2.iterrows():
            for dk in range(-4, 1):
                k = int(r.start) + dk
                if 0 <= k < H:
                    rel.setdefault(dk, []).append((r[cols[k]], st[cols[k]].mean() if len(st) else np.nan))
        print("    sigma_p aligned to the start (k - k*): starters / stayers at the same steps (n)")
        print("    " + "  ".join(f"{dk:+d}: {np.mean([a for a, _ in v]):.3f}/{np.nanmean([b for _, b in v]):.3f} "
                                  f"({len(v)})" for dk, v in sorted(rel.items())))
        row["aligned"] = {dk: (float(np.mean([a for a, _ in v])), float(np.nanmean([b for _, b in v])), len(v))
                          for dk, v in rel.items()}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_windows", type=int, default=3000)
    p.add_argument("--horizon", type=int, default=10, help="steps for 'starts soon' and for sigma_p (10 = 2 s)")
    p.add_argument("--n_future", type=int, default=20, help="steps for 'stays' (20 = 4 s; <= the cache's n_pred)")
    p.add_argument("--min_area", type=int, default=6)
    p.add_argument("--region", type=float, nargs=3, default=None, metavar=("X", "Y", "R"))
    p.add_argument("--k", type=int, default=0, help="prior rollouts per window: does the model IMAGINE the start?")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    n, H, rc = cfg.data.n_cond, a.horizon, cfg.render
    assert a.horizon <= a.n_future <= cfg.data.n_pred
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.ckpts[0]))
    os.makedirs(out_dir, exist_ok=True)
    curves = {}
    for path in a.ckpts:
        s = torch.load(path, map_location=dev)
        m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
        m.load_state_dict(s["model"])
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if m.cfg.deterministic or m.cfg.z_global or m.cfg.z_seq:
            print(f"\n{run}: no spatial latent, skipped"); continue
        ds = RounDWindows(a.cache, split, cfg, s["stats"], n_pred=cfg.data.n_pred)
        idx = np.sort(np.random.default_rng(a.seed).choice(len(ds), min(a.n_windows, len(ds)), replace=False))
        rows = []
        for j in range(0, len(idx), a.batch):
            items = [ds[int(i)] for i in idx[j:j + a.batch]]
            b = {key: torch.stack([it[key] for it in items]).to(dev) for key in items[0]}
            torch.manual_seed(a.seed + j)
            sig, kl = posterior_track(m, b, n, H)                                       # (B, H, 8, 8)
            up = lambda x: x.repeat_interleave(8, -1).repeat_interleave(8, -2)            # -> (B, H, 64, 64)
            sig, kl = up(sig), up(kl)
            if a.k:
                from ladder import scheduled_rollout
                gs = torch.stack([scheduled_rollout(m, b, n, H, lambda q: "prior") for _ in range(a.k)], 1)
            green, pose = b["images"][:, :, GREEN].float(), b["ego_state"][..., :3]
            for bi in range(len(items)):
                for u in stopped_users(green[bi], pose[bi], n, a.n_future, rc, a.min_area):
                    if u["start"] is None:
                        continue
                    k0 = u["start"]
                    label = "stays" if k0 < 0 else ("start_2s" if k0 < H else "start_4s")
                    ys, xs = np.nonzero(u["mask"].cpu().numpy())
                    fwd = (rc.ego_row - ys.mean()) * rc.m_per_px
                    left = (rc.ego_col - xs.mean()) * rc.m_per_px
                    ex, ey, eh = (float(v) for v in pose[bi, n - 1])
                    r = {"win": int(idx[j + bi]), "area": u["area"], "start": k0, "label": label,
                         "wx": ex + fwd * np.cos(eh) - left * np.sin(eh),
                         "wy": ey + fwd * np.sin(eh) + left * np.cos(eh)}
                    for k in range(H):
                        f = u["fp"][k]
                        r[f"sig_{k}"] = float(sig[bi, k][f].mean()) if f.any() else np.nan
                        r[f"kl_{k}"] = float(kl[bi, k][f].mean()) if f.any() else np.nan
                    if a.k:
                        res = [sample_start(gs[bi, q], u) for q in range(a.k)]
                        ks = [q for q, fd in res if q >= 0 and not fd]
                        r.update(k_samples=a.k, p_start=len(ks) / a.k,
                                 p_fade=sum(1 for q, fd in res if q >= 0 and fd) / a.k,
                                 k_pred=float(np.mean(ks)) if ks else -1.0)
                    rows.append(r)
        d = pd.DataFrame(rows)
        print(f"\n{run}: {len(idx)} windows, horizon {H * 0.2:.1f} s, 'stays' = still there after "
              f"{a.n_future * 0.2:.1f} s")
        if d.empty:
            print("  no stopped road users found"); continue
        d.to_csv(os.path.join(out_dir, f"yield_starts_{run}.csv"), index=False)
        row = report(d, H, "all")
        s2 = d[d.label == "start_2s"]
        if len(s2):
            gx, gy = (s2.wx // 6 * 6 + 3).astype(int), (s2.wy // 6 * 6 + 3).astype(int)
            top = pd.Series(list(zip(gx, gy))).value_counts().head(6)
            print("    where the starters were (6 m bins): " + ", ".join(f"{xy} n{c}" for xy, c in top.items()))
        if a.region:
            X, Y, R = a.region
            report(d[np.hypot(d.wx - X, d.wy - Y) <= R], H, f"within {R:.0f} m of ({X:.0f}, {Y:.0f})")
        curves[run] = row.get("aligned", {})
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (run, al) in enumerate(curves.items()):
            if not al:
                continue
            ks = sorted(al)
            c = f"C{i}"
            ax.plot(ks, [al[k][0] for k in ks], "-o", color=c, lw=2, ms=5, label=f"{run}: starters")
            ax.plot(ks, [al[k][1] for k in ks], "--", color=c, lw=2, label=f"{run}: stayers, same steps")
        ax.set_xlabel("steps relative to the start (0 = first frame where it has moved)")
        ax.set_ylabel("prior sigma over the footprint")
        ax.set_title("Does the prior widen before a stopped road user goes?", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "yield_starts.png"), dpi=120)
        print(f"\nwrote yield_starts_<run>.csv and yield_starts.png to {out_dir}")


if __name__ == "__main__":
    main()
