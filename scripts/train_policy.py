"""Train the learned policy (controller B) through a frozen world model, PPUU / MPUR-style.

    python scripts/train_policy.py --cache ../round_cache --wm runs/ab_inv4_01/last.pt --name pol_v4 \
        --steps 10000 --lam 0.5

See roundabout_wm/controller/policy.py for the policy, the rollout and the cost. Per step: a batch of train-split
windows, h warmed on their 10 real frames, T imagined steps with the policy acting on its own imagined h, one prior
sample per step, loss = mean over the batch of the total cost; only the policy's weights are trained.

Before training (if --lam > 0 and --masks > 0): the uncertainty cost's normalisation, the mean and std per step of
U under the human's own logged accelerations on --calib_windows train windows (PPUU normalises U with training-set
statistics). Every --val_every steps: mean costs on --n_val fixed val windows (no gradient); best.pt = lowest val
total. Writes runs/<name>/{best,last}.pt (policy weights, its config, the world model path, the cost weights, the
U normalisation) and log.csv.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.policy import PolicyNet, policy_rollout, rollout_cost, routes_from_batch
from roundabout_wm.dataset import RounDWindows
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel


def load_wm(path, dev):
    s = torch.load(path, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    m.requires_grad_(False)
    return m, s["stats"]


@torch.no_grad()
def warm(m, b, n):
    h, _ = m.warmup(m.encode(b["images"][:, :n]), b["actions"], b["speed"])
    return h, b["images"][:, n - 1, GREEN:GREEN + 1]


def batches(ds, idx, bs, dev, rng=None, loop=False, workers=0):
    """Batches of the windows idx (dicts of images, actions, speed, ego_state on dev). loop: shuffled, forever."""
    from torch.utils.data import DataLoader, Subset
    keys = ("images", "actions", "speed", "ego_state")
    while True:
        order = rng.permutation(idx) if rng is not None else np.asarray(idx)
        dl = DataLoader(Subset(ds, [int(i) for i in order]), batch_size=bs, shuffle=False, drop_last=loop,
                        num_workers=workers, persistent_workers=False)
        for b in dl:
            yield {k: b[k].to(dev, non_blocking=True) for k in keys}
        if not loop:
            return


def rollout_batch(m, pi, b, stats, cfg, T, masks, eps=None, accels=None):
    n = cfg.data.n_cond
    h0, prev0 = warm(m, b, n)
    v0, rs, rk, acc_prev, logged = routes_from_batch(b, stats, cfg)
    out = policy_rollout(m, h0, prev0, v0, rs, rk, stats, cfg, T, policy=pi,
                         accels=logged[:, :T] if accels == "logged" else None, eps=eps, n_masks=masks)
    return out, acc_prev


@torch.no_grad()
def calibrate_uncertainty(m, ds, idx, stats, cfg, T, masks, bs, dev):
    """Mean and std per step of U under the human's own accelerations."""
    us = []
    for b in batches(ds, idx, bs, dev):
        out, _ = rollout_batch(m, None, b, stats, cfg, T, masks, accels="logged")
        us.append(out["unc"].cpu())
    u = torch.cat(us)
    return u.mean(0).numpy(), u.std(0).clamp(min=1e-6).numpy()


@torch.no_grad()
def validate(m, pi, ds, idx, stats, cfg, T, masks, bs, dev, cc, unc_norm, lam, seed):
    pi.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = {}
    nb = 0
    for b in batches(ds, idx, bs, dev):
        eps = [torch.randn((len(b["images"]), m.cfg.z_ch, 8, 8), generator=g, device=dev) for _ in range(T)]
        out, acc_prev = rollout_batch(m, pi, b, stats, cfg, T, masks if lam > 0 else 0, eps=eps)
        c = rollout_cost(out, cfg, cc, acc_prev, unc_norm, lam)
        c["acc_mean"] = out["acc"].mean(1)
        c["v_end"] = out["v"][:, -1]
        for k, v in c.items():
            acc[k] = acc.get(k, 0.0) + float(v.mean())
        nb += 1
    pi.train()
    return {k: v / max(nb, 1) for k, v in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--wm", required=True, help="world model checkpoint (frozen)")
    p.add_argument("--name", required=True)
    p.add_argument("--out_root", default="runs")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=32, help="windows per step (K=4 masks: ~0.72 GB activations each)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--T", type=int, default=10, help="imagined steps per rollout (10 = 2 s)")
    p.add_argument("--masks", type=int, default=4, help="dropout masks K for the uncertainty cost")
    p.add_argument("--lam", type=float, default=0.5, help="uncertainty cost weight (0 = off)")
    p.add_argument("--w_prog", type=float, default=0.5, help="progress weight (planner's calibrated value)")
    p.add_argument("--calib_windows", type=int, default=512)
    p.add_argument("--n_val", type=int, default=512)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--workers", type=int, default=8, help="data loader workers for the training batches")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cfg = Config()
    out_dir = os.path.join(a.out_root, a.name)
    os.makedirs(out_dir, exist_ok=True)
    m, stats = load_wm(a.wm, dev)
    split = json.load(open(os.path.join(a.cache, "split.json")))
    tr = RounDWindows(a.cache, split["train"], cfg, stats, n_pred=cfg.data.n_pred)
    va = RounDWindows(a.cache, split["val"], cfg, stats, n_pred=cfg.data.n_pred)
    val_idx = np.sort(np.random.default_rng(a.seed + 1).choice(len(va), min(a.n_val, len(va)), replace=False))
    cc = CostConfig(w_prog=a.w_prog)
    masks = a.masks if a.lam > 0 else 0
    unc_norm = None
    if masks:
        cal_idx = rng.choice(len(tr), min(a.calib_windows, len(tr)), replace=False)
        unc_norm = calibrate_uncertainty(m, tr, cal_idx, stats, cfg, a.T, masks, a.batch, dev)
        print("uncertainty normalisation (human actions), per step: mean "
              + " ".join(f"{x:.3g}" for x in unc_norm[0]) + " | std " + " ".join(f"{x:.3g}" for x in unc_norm[1]))
    pi = PolicyNet(hidden=m.cfg.hidden).to(dev)
    opt = torch.optim.Adam(pi.parameters(), lr=a.lr)
    meta = dict(wm=os.path.abspath(a.wm), args=vars(a), cost=cc.to_dict(),
                unc_norm=None if unc_norm is None else [x.tolist() for x in unc_norm])
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(meta, f, indent=1)
    fields = ["step", "split", "total", "risk", "progress", "comfort", "unc", "collision", "proximity", "acc_mean",
              "v_end", "sec"]
    log_f = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=fields, extrasaction="ignore")
    log.writeheader()
    save = lambda path, step: torch.save({"policy": pi.state_dict(), "step": step, **meta}, path)
    best, t0 = np.inf, time.time()
    it = batches(tr, np.arange(len(tr)), a.batch, dev, rng=rng, loop=True, workers=a.workers)
    run = {}
    for step in range(1, a.steps + 1):
        b = next(it)
        out, acc_prev = rollout_batch(m, pi, b, stats, cfg, a.T, masks)
        c = rollout_cost(out, cfg, cc, acc_prev, unc_norm, a.lam)
        loss = c["total"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        c["acc_mean"], c["v_end"] = out["acc"].mean(1), out["v"][:, -1]
        for k, v in c.items():
            run[k] = run.get(k, 0.0) + float(v.detach().mean()) / a.log_every
        if step % a.log_every == 0:
            row = dict(step=step, split="train", sec=round(time.time() - t0), **run)
            log.writerow(row); log_f.flush()
            print(f"step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in run.items()), flush=True)
            run = {}
        if step % a.val_every == 0 or step == a.steps:
            v = validate(m, pi, va, val_idx, stats, cfg, a.T, masks, a.batch, dev, cc, unc_norm, a.lam, a.seed)
            log.writerow(dict(step=step, split="val", sec=round(time.time() - t0), **v)); log_f.flush()
            print(f"  VAL step {step}: " + " ".join(f"{k} {x:.4g}" for k, x in v.items()), flush=True)
            if v["total"] < best:
                best = v["total"]
                save(os.path.join(out_dir, "best.pt"), step)
    save(os.path.join(out_dir, "last.pt"), a.steps)
    print(f"done: best val total {best:.4g}; wrote {out_dir}")


if __name__ == "__main__":
    main()
