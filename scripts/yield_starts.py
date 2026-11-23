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
   ego poses. It has "started" at the first k where > --moved (default 30 %) of the footprint is no longer green
   (0.3: a car has moved ~1.4 m, so creeping forward in a queue counts; 0.8: about a car length). Labels:
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
    - with --masks M, M MC-dropout rollouts (the same prior noise, only the dropout mask differs): the same shares
      for the masks, how often the masks disagree, and (with --k) whether mask disagreement flags the road users on
      which the latent samples all agree and are wrong: can an epistemic penalty catch confident mistakes?
    - with --flip R, the single-latent test: a reference rollout (noise e0), a full resample (e1), and for every
      step k and scope (all cells / the stopped road users' cells within 4.5 m / every other cell) the reference with
      ONLY that latent resampled from e1. Prints, per step and scope, the share of stopped road users whose go/wait
      outcome flips, and writes flip_<run>.png (the flips with the largest change in how empty the car's spot is at 2 s, one per window:
      reference vs flipped, |difference|), and how big the flips are (a flip from 0.28 to 0.32 empty is a nudge
      across the threshold, not a decision). Flips at step
      1-2 = an early latent tips the decision and the state keeps it; one step's flips equal to
      the full resample = that step decides; flips spread over steps summing to ~sqrt(steps) x the full resample =
      accumulated; flips from "elsewhere" = the decision depends on other cells (e.g. the
      ring traffic), not on the car's own latent. Cells of nearby stopped road users share the "its cells" scope.
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
import torch.nn.functional as F

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel
from roundabout_wm.wm_eval import warp_previous
from uncertainty_map import dropout_rollouts, posterior_track


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


@torch.no_grad()
def rollout_eps(m, b, n, H, eps):
    """Prior rollout with GIVEN standard-normal noise: z_k = mu_p(h_k) + sigma_p(h_k) * eps[k], eps (H, B, z_ch, 8, 8)
    (the reparameterisation, so swapping eps[k] resamples exactly the latent of step k). Eval mode. -> (B, H, 64, 64)"""
    imgs, act, spd = b["images"], b["actions"], b["speed"]
    was = m.training
    m.eval()
    try:
        e = m.encode(imgs[:, :n])
        h, _ = m.warmup(e, act, spd)
        prev = imgs[:, n - 1, GREEN:GREEN + 1]
        out = []
        for k in range(H):
            i = n - 1 + k
            frame = None
            if m.cfg.reencode:
                frame = imgs[:, i] if k == 0 else torch.cat([imgs[:, i, :GREEN], prev.to(imgs.dtype),
                                                             imgs[:, i, GREEN + 1:]], 1)
            mu, lv = m.prior_params(h)
            h, prev = m.rollout_step(h, mu + (0.5 * lv).exp() * eps[k], act[:, i], spd[:, i], frame, prev)
            out.append(prev[:, 0].float())
    finally:
        m.train(was)
    return torch.stack(out, 1)


SCOPES = ("all", "car", "other")


def flip_rollouts(m, b, n, H, cellmask, gen):
    """One reference rollout with noise e0, one full resample (e1 everywhere), and for every step k and scope
    (all cells / the stopped road users' cells / every other cell) the reference with ONLY that step's z, in that
    scope, resampled from e1. cellmask (B, H, 1, 8, 8) bool: cells within 4.5 m of a stopped road user at step k.
    -> dict {"ref": g, "full": g, (scope, k): g}, each g (B, H, 64, 64)."""
    B, C = b["images"].shape[0], m.cfg.z_ch
    dev = b["images"].device
    e0 = torch.randn((H, B, C, 8, 8), generator=gen, device=dev)
    e1 = torch.randn((H, B, C, 8, 8), generator=gen, device=dev)
    out = {"ref": rollout_eps(m, b, n, H, e0), "full": rollout_eps(m, b, n, H, e1)}
    for k in range(H):
        for sc in SCOPES:
            e = e0.clone()
            sel = torch.ones_like(cellmask[:, k]) if sc == "all" else (cellmask[:, k] if sc == "car" else ~cellmask[:, k])
            e[k] = torch.where(sel, e1[k], e0[k])
            out[(sc, k)] = rollout_eps(m, b, n, H, e)
    return out


def went(g, u, moved=0.3):
    k, faded = sample_start(g, u, moved)
    return k >= 0 and not faded


def empty_frac(g_last, u, H):
    """Share of the road user's footprint (warped to the last step) that is empty in the frame g_last (64, 64):
    the continuous quantity the 'drives off' threshold cuts (0 = still there, 1 = gone from its spot)."""
    f = u["fp"][H - 1]
    return float((g_last[f] < 0.5).float().mean()) if f.any() else float("nan")


def flip_report(d, H):
    """Which single latent decides whether a stopped road user goes?"""
    print(f"    single-latent flips ({int(d.n_flip.iloc[0])} reference/alternative draws per window): share of "
          f"stopped road users whose go/wait outcome changes vs the reference")
    print(f"      every latent resampled (all steps, all cells): {d.flip_full.mean():.3f}")
    print(f"      {'step':>4s} {'all cells':>10s} {'its cells':>10s} {'elsewhere':>10s}")
    for k in range(H):
        print(f"      {k + 1:4d} " + " ".join(f"{d[f'flip_{sc}_{k}'].mean():10.3f}" for sc in SCOPES))
    tot = {sc: sum(d[f"flip_{sc}_{k}"].mean() for k in range(H)) for sc in SCOPES}
    full = max(float(d.flip_full.mean()), 1e-9)
    peak = int(np.argmax([d[f"flip_all_{k}"].mean() for k in range(H)])) + 1
    print("      sum over steps: " + ", ".join(f"{sc} {v:.3f}" for sc, v in tot.items())
          + f"; sum / full resample = {tot['all'] / full:.1f}, peak at step {peak}")
    print("      reading: one step decides -> its flips equal the full resample and the others are ~0 (ratio ~1); a\n"
          "      threshold on a SUM of small per-step contributions -> flips spread over steps and the ratio is ~sqrt(steps)\n"
          "      (> 1: one swap can tip it, swapping all tends to cancel); tipped early and kept -> peak at step 1-2")
    if "p_start" in d:
        c = d[(d.p_start > 0) & (d.p_start < 1)]
        if len(c):
            print(f"      cars whose 8 samples split ({len(c)}): full {c.flip_full.mean():.3f}; by step, all cells: "
                  + " ".join(f"{c[f'flip_all_{k}'].mean():.2f}" for k in range(H)))
    return {"flip_full": float(d.flip_full.mean()), **{f"flip_sum_{sc}": v for sc, v in tot.items()}}


def flip_sheet(ex, run, path):
    """Examples where resampling ONE latent flipped go/wait: reference vs flipped at 2 s, and |difference|."""
    if not ex:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(ex), 3, figsize=(7.5, 2.5 * len(ex)), squeeze=False)
    for i, (ref, flp, fp, lab) in enumerate(ex):
        for j, (img, t, cm) in enumerate(((ref, "reference", "gray"), (flp, "one latent resampled", "gray"),
                                          (np.abs(flp - ref), "|difference|", "magma"))):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=0, vmax=1)
            ax.contour(fp, levels=[0.5], colors="c", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(t, fontsize=8)
        axes[i, 0].set_ylabel(lab, fontsize=7)
    fig.suptitle(f"{run}: one latent decides go / wait (green channel at 2 s; cyan = the stopped road user's spot; "
                 f"ego forward is up)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"    wrote {path}")


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
    if "p_mask" in d:
        print(f"    dropout ({int(d.n_masks.iloc[0])} masks, same prior noise): share of masks where it drives off / "
              f"vanishes")
        for lab, g in (("start within 2 s", s2), ("start 2-4 s", s4), ("stays", st)):
            if len(g):
                print(f"      truly {lab:17s} drives off {g.p_mask.mean():.2f}   vanishes {g.p_mask_fade.mean():.2f}")
        go = (d.label == "start_2s").astype(int)
        msplit = d.p_mask * (1 - d.p_mask)
        print(f"    masks disagree (not all the same) on {(msplit > 0).mean():.2f} of stopped road users; AUROC of the "
              f"mask share, start_2s vs stays {auroc(s2.p_mask, st.p_mask):.2f}")
        row["auroc_masks"] = auroc(s2.p_mask, st.p_mask)
        if "p_start" in d:
            lsplit = d.p_start * (1 - d.p_start)
            wrong = ((d.p_start > 0.5).astype(int) != go)
            conf = (d.p_start <= 1 / 8) | (d.p_start >= 7 / 8)
            rho = lambda x, y: pd.concat([x, y], axis=1).corr(method="spearman").iloc[0, 1]
            print(f"    Spearman mask disagreement vs latent disagreement {rho(msplit, lsplit):.2f}; vs being wrong "
                  f"(latent majority) {rho(msplit, wrong.astype(float)):.2f}")
            cw, cr = msplit[conf & wrong], msplit[conf & ~wrong]
            print(f"    the latent samples all agree on {conf.mean():.2f} of them and are wrong on {(conf & wrong).sum()}"
                  f": do the masks flag those? mask disagreement wrong {cw.mean():.3f} vs right {cr.mean():.3f}, "
                  f"AUROC {auroc(cw, cr):.2f}")
            row["auroc_masks_flag_confident_wrong"] = auroc(cw, cr)
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
    p.add_argument("--masks", type=int, default=0, help="MC-dropout rollouts per window (same prior noise, 0 = off)")
    p.add_argument("--flip", type=int, default=0, help="single-latent resampling test: draws per window (0 = off)")
    p.add_argument("--flip_sheet", type=int, default=8, help="flip examples drawn in flip_<run>.png (largest changes)")
    p.add_argument("--moved", type=float, default=0.3,
                   help="'started / drives off' = more than this share of the footprint empty (0.3 ~ 1.4 m of "
                        "movement; ~0.8 ~ a car length). Applies to the true labels and to the imagined futures")
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
        rows, flip_cands, flip_gap = [], [], []
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
            if a.masks:
                gd = dropout_rollouts(m, b, n, H, a.masks).transpose(0, 1)               # (B, M, H, 64, 64)
            green, pose = b["images"][:, :, GREEN].float(), b["ego_state"][..., :3]
            users = [[u for u in stopped_users(green[bi], pose[bi], n, a.n_future, rc, a.min_area,
                                                  moved=a.moved)
                      if u["start"] is not None] for bi in range(len(items))]
            flips = []
            if a.flip:
                cellmask = torch.zeros(len(items), H, 1, 8, 8, dtype=torch.bool, device=dev)
                for bi, us in enumerate(users):
                    for u in us:
                        cellmask[bi, :, 0] |= F.max_pool2d(u["near"][:H].float()[:, None], 8)[:, 0] > 0
                gen = torch.Generator(device=dev).manual_seed(a.seed + j + 7)
                if any(users):
                    flips = [flip_rollouts(m, b, n, H, cellmask, gen) for _ in range(a.flip)]
            for bi in range(len(items)):
                for u in users[bi]:
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
                        res = [sample_start(gs[bi, q], u, a.moved) for q in range(a.k)]
                        ks = [q for q, fd in res if q >= 0 and not fd]
                        r.update(k_samples=a.k, p_start=len(ks) / a.k,
                                 p_fade=sum(1 for q, fd in res if q >= 0 and fd) / a.k,
                                 k_pred=float(np.mean(ks)) if ks else -1.0)
                    if a.masks:
                        res = [sample_start(gd[bi, q], u, a.moved) for q in range(a.masks)]
                        r.update(n_masks=a.masks,
                                 p_mask=sum(1 for q, fd in res if q >= 0 and not fd) / a.masks,
                                 p_mask_fade=sum(1 for q, fd in res if q >= 0 and fd) / a.masks)
                    if flips:
                        r["n_flip"] = a.flip
                        acc = {"full": 0.0, **{f"{sc}_{k}": 0.0 for sc in SCOPES for k in range(H)}}
                        for fr in flips:
                            w0 = went(fr["ref"][bi], u, a.moved)
                            e0 = empty_frac(fr["ref"][bi, H - 1], u, H)
                            acc["full"] += (went(fr["full"][bi], u, a.moved) != w0) / a.flip
                            for sc in SCOPES:
                                for k in range(H):
                                    f = went(fr[(sc, k)][bi], u, a.moved) != w0
                                    acc[f"{sc}_{k}"] += f / a.flip
                                    if f:
                                        e1 = empty_frac(fr[(sc, k)][bi, H - 1], u, H)
                                        flip_gap.append(abs(e1 - e0))
                                        if sc != "all" and np.isfinite(e1 - e0):
                                            flip_cands.append((abs(e1 - e0), int(idx[j + bi]), (
                                                fr["ref"][bi, H - 1].cpu().numpy(),
                                                fr[(sc, k)][bi, H - 1].cpu().numpy(),
                                                u["fp"][H - 1].float().cpu().numpy(),
                                                f"win {int(idx[j + bi])}: step {k + 1}, "
                                                f"{'its cells' if sc == 'car' else 'elsewhere'}; ref "
                                                f"{'goes' if w0 else 'waits'}; spot empty at 2 s: ref {e0:.2f}, "
                                                f"resampled {e1:.2f}")))
                                            flip_cands.sort(key=lambda c: -c[0])
                                            seen, keep = set(), []
                                            for c in flip_cands:              # largest changes, one per window
                                                if c[1] not in seen:
                                                    seen.add(c[1]); keep.append(c)
                                            flip_cands[:] = keep[:a.flip_sheet]
                        r.update({f"flip_{key}": v for key, v in acc.items()})
                    rows.append(r)
        d = pd.DataFrame(rows)
        print(f"\n{run}: {len(idx)} windows, horizon {H * 0.2:.1f} s, 'stays' = still there after "
              f"{a.n_future * 0.2:.1f} s; 'starts' = more than {a.moved:.0%} of its spot empty")
        if d.empty:
            print("  no stopped road users found"); continue
        tag = "" if a.moved == 0.3 else f"_m{a.moved:g}"
        d.to_csv(os.path.join(out_dir, f"yield_starts_{run}{tag}.csv"), index=False)
        row = report(d, H, "all")
        if "flip_full" in d:
            row.update(flip_report(d, H))
            if flip_gap:
                g0 = np.array(flip_gap)
                g = g0[np.isfinite(g0)]                       # nan: the spot has left the view by 2 s
                if len(g):
                    print(f"      how big are the flips? change in the share of the spot that is empty at 2 s, over "
                          f"{len(g)} flips ({len(g0) - len(g)} more with the spot out of view at 2 s): median "
                          f"{np.median(g):.2f}, p90 {np.percentile(g, 90):.2f}; share of flips with a change > 0.5 "
                          f"(a real drive-off vs wait): {(g > 0.5).mean():.2f}")
                    row["flip_big_share"] = float((g > 0.5).mean())
            flip_sheet([c[2] for c in flip_cands], run, os.path.join(out_dir, f"flip_{run}{tag}.png"))
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
