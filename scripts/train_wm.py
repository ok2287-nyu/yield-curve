"""Train world model v1. The three team runs differ by flags only (same code, data, seed, steps):

    python scripts/train_wm.py --cache $C --encoder cnn --run cnn
    python scripts/train_wm.py --cache $C --encoder vit --run vit
    python scripts/train_wm.py --cache $C --encoder vit --init_encoder runs/mae/encoder.pt --run vit_mae
    python scripts/train_wm.py --cache $C --encoder cnn --run cnn_nosg --no-prior_sg     # stop-gradient ablation

Rerunning the same command resumes from runs/<run>/last.pt (written every --ckpt_minutes), which is what makes
#SBATCH --requeue and the Cloud Bursting idle shutdown survivable.

Validation (val recordings, dropout OFF), all per rollout step, summarised at 1 s / 2 s / 4 s (steps 5/10/20):
    mse_prior    open-loop MSE with z from the learned PRIOR: the number that matters
    mse_post     the same with z from the posterior; gap = prior - post (large and persistent = posterior shortcut)
    empty/copy/warp  baselines: predict nothing; repeat the last observed frame; warp it by the logged ego motion.
                 The gate: beat empty and warp at 1, 2 and 4 s.
    kl_t         KL(q || p) per step
    prior_std_t  mean prior sigma per step (flat or shrinking towards 0 = deterministic prior)
    mass_t       predicted / true green mass per step (falling = cars fading)
    act_sens     mean |change| of the prediction when the future actions are swapped between samples
                 (speeds re-integrated from the swapped accelerations); ~0 = the model ignores the ego
Every validation also writes samples_<step>.png: true future, 3 prior samples, and the prior sigma map.

Weights & Biases (optional, --wandb): the same train/val numbers as log.csv, the per-step error curves
(model vs baselines over the whole 4 s), and the sample grids. log.csv stays the source of truth. The W&B run id is
kept in runs/<run>/wandb_id.txt, so a requeued job continues the same W&B run. If the compute node has no
internet, run with WANDB_MODE=offline and later `wandb sync runs/<run>/wandb/offline-run-*` from a login node.
"""
import argparse
import csv
import json
import os
import secrets
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows, load_stats
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel, param_counts
from roundabout_wm.wm_eval import swap_actions, warp_previous

REPORT_STEPS = (5, 10, 20)
VAL_SEED = 1234                     # fixes which val windows are used; reseeded before every validation          # rollout steps reported in the logs (1 s, 2 s, 4 s at 5 Hz)


def args_():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--run", required=True, help="run name; outputs in <runs_dir>/<run>")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--encoder", default="cnn", choices=["cnn", "vit"])
    p.add_argument("--init_encoder", default=None, help="MAE-pretrained encoder weights (vit only)")
    p.add_argument("--enc_lr_mult", type=float, default=None,
                   help="encoder lr multiplier; default 0.1 with --init_encoder, else 1.0")
    p.add_argument("--n_pred", type=int, default=cfg.data.n_pred)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1e-6)
    p.add_argument("--p_zdrop", type=float, default=0.5)
    p.add_argument("--z_ch", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prior_sg", action=argparse.BooleanOptionalAction, default=True,
                   help="stop-gradient on prior samples in z-dropout training (--no-prior_sg for the ablation)")
    p.add_argument("--steps", type=int, default=60000, help="set after timing the first run; same for all runs")
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--ckpt_minutes", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_windows", type=int, default=None, help="debug: use only the first N windows per split")
    p.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "yield-curve"))
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"), help="W&B team/user; shared team = one place")
    a = p.parse_args()
    if a.enc_lr_mult is None:
        a.enc_lr_mult = 0.1 if a.init_encoder else 1.0
    return a


def loaders(a, cfg):
    split_path = os.path.join(a.cache, "split.json")
    if os.path.exists(split_path):
        split = json.load(open(split_path))
    else:                                    # debugging on a partial cache: train and validate on everything
        recs = sorted(int(d[4:]) for d in os.listdir(a.cache) if d.startswith("rec_"))
        split = {"train": recs, "val": recs}
        print("WARNING: no split.json, validating on training recordings")
    stats_path = os.path.join(a.cache, "stats.json")
    if os.path.exists(stats_path):
        stats = load_stats(stats_path)
    else:
        print("WARNING: no stats.json, inputs are NOT normalised")
        stats = None
    tr = RounDWindows(a.cache, split["train"], cfg, stats, n_pred=a.n_pred)
    va = RounDWindows(a.cache, split["val"], cfg, stats, n_pred=a.n_pred)
    if a.max_windows:
        tr, va = Subset(tr, range(min(a.max_windows, len(tr)))), Subset(va, range(min(a.max_windows, len(va))))
    g = torch.Generator().manual_seed(a.seed)
    kw = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=True, drop_last=True,
              persistent_workers=a.workers > 0)
    # a fixed generator for validation: every validation sees the same windows in the same order
    val_loader = DataLoader(va, shuffle=True, generator=torch.Generator().manual_seed(VAL_SEED), **kw)
    stats = stats or {"speed_mean": 0.0, "speed_std": 1.0, "action_mean": [0.0, 0.0], "action_std": [1.0, 1.0]}
    return DataLoader(tr, shuffle=True, generator=g, **kw), val_loader, len(tr), len(va), stats


def to(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def per_step(err):
    """(B, T, 1, H, W) squared error -> (T,) mean over batch and pixels."""
    return err.float().mean((0, 2, 3, 4))


@torch.no_grad()
def validate(model, val_loader, a, dev, cfg, stats):
    """All metrics per rollout step, z from the prior unless stated, dropout off."""
    model.eval()
    val_loader.generator.manual_seed(VAL_SEED)       # the SAME val windows every validation (comparable over time)
    n, rc, dt = cfg.data.n_cond, cfg.render, cfg.data.dt
    acc = {k: 0 for k in ("prior", "post", "empty", "copy", "warp", "kl", "std", "mass_pred", "mass_true")}
    sens, warm, nb = 0.0, 0.0, 0
    for i, b in enumerate(val_loader):
        if i >= a.val_batches:
            break
        b = to(b, dev)
        imgs, act, spd, st = b["images"], b["actions"], b["speed"], b["ego_state"]
        tgt = imgs[:, n:n + a.n_pred, GREEN:GREEN + 1]

        torch.manual_seed(1234 + i)
        out = model(imgs, act, spd, a.n_pred, z_source="prior")
        torch.manual_seed(1234 + i)                                   # same noise, other sample's future actions
        act_sw, spd_sw = swap_actions(act, spd, st, n, stats, dt)
        out_sw = model(imgs, act_sw, spd_sw, a.n_pred, z_source="prior")
        post = model(imgs, act, spd, a.n_pred, z_source="post")

        last = imgs[:, n - 1, GREEN:GREEN + 1]
        warp = warp_previous(last, st[:, n - 1, :3], st[:, n:n + a.n_pred, :3], rc)
        acc["prior"] = acc["prior"] + per_step((out["green"] - tgt) ** 2)
        acc["post"] = acc["post"] + per_step((post["green"] - tgt) ** 2)
        acc["empty"] = acc["empty"] + per_step(tgt ** 2)
        acc["copy"] = acc["copy"] + per_step((last[:, None] - tgt) ** 2)
        acc["warp"] = acc["warp"] + per_step((warp - tgt) ** 2)
        acc["kl"] = acc["kl"] + post["kl"].mean(0)
        acc["std"] = acc["std"] + out["prior_std"].mean(0)
        acc["mass_pred"] = acc["mass_pred"] + out["green"].float().sum((2, 3, 4)).mean(0)
        acc["mass_true"] = acc["mass_true"] + tgt.sum((2, 3, 4)).mean(0)
        sens += (out["green"] - out_sw["green"]).abs().float().mean().item()
        warm += ((out["green_warm"].float() - imgs[:, :n, GREEN:GREEN + 1]) ** 2).mean().item()
        nb += 1
    model.train()
    v = {k: (x / nb).cpu() for k, x in acc.items()}
    v["mass"] = v.pop("mass_pred") / v.pop("mass_true").clamp_min(1e-8)
    v["gap"] = v["prior"] - v["post"]
    v["act_sens"], v["mse_warm"] = sens / nb, warm / nb
    return v


@torch.no_grad()
def save_samples(model, batch, a, dev, n_cond, path, n_samples=3):
    """One validation window. Rows: true future; prior samples 1..3 (true road/ego + predicted green);
    prior sigma per cell (mean over z channels), drawn as grey = sigma / 1.5 (white = sigma >= 1.5)."""
    model.eval()
    b = {k: v[:1].to(dev) for k, v in batch.items()}
    imgs, act, spd = b["images"], b["actions"], b["speed"]
    truth = imgs[0, n_cond:n_cond + a.n_pred]
    rows = [truth]
    for s in range(n_samples):
        torch.manual_seed(s)
        out = model(imgs, act, spd, a.n_pred, z_source="prior")
        pred = truth.clone()
        pred[:, GREEN:GREEN + 1] = out["green"][0].float()
        rows.append(pred)
    sig = (out["prior_std_map"][0] / 1.5).clamp(0, 1)                         # (T, 8, 8), last sample's prior
    sig = torch.nn.functional.interpolate(sig[:, None], scale_factor=8, mode="nearest").expand(-1, 3, -1, -1)
    rows.append(sig.to(truth.dtype))
    step = max(1, a.n_pred // 10)
    grid = torch.cat([torch.cat(list(r[step - 1::step]), 2) for r in rows], 1)   # columns = steps step, 2*step, ...
    Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)) \
        .resize((grid.shape[2] * 3, grid.shape[1] * 3), Image.NEAREST).save(path)
    model.train()


def gpu_util():
    """Instantaneous GPU utilisation in % from nvidia-smi (no pynvml needed); "" if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else ""
    except Exception:
        return ""


LOG_FIELDS = (["step", "split", "loss", "mse", "mse_warm", "kl", "prior_std", "act_sens", "gap_mean", "s_per_step",
               "gpu_util", "time"]
              + [f"{m}@{k}" for k in REPORT_STEPS for m in ("prior", "post", "empty", "copy", "warp", "mass", "kl",
                                                             "std")])


class WandB:
    """Thin optional wrapper: every method is a no-op when W&B is off, and a W&B failure never stops training."""

    def __init__(self, a, out_dir, extra_config):
        self.run = None
        if not a.wandb:
            return
        try:
            import wandb
            id_path = os.path.join(out_dir, "wandb_id.txt")
            run_id = open(id_path).read().strip() if os.path.exists(id_path) else secrets.token_hex(6)
            open(id_path, "w").write(run_id)
            self.wandb = wandb
            self.run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=a.run, id=run_id,
                                  resume="allow", dir=out_dir, config={**vars(a), **extra_config},
                                  group=a.encoder, job_type="train_wm")
            print(f"W&B: {self.run.url}")
        except Exception as e:                                           # no login, no network, not installed ...
            print(f"WARNING: W&B disabled ({type(e).__name__}: {e}); log.csv is still written")
            self.run = None

    def log(self, d, step):
        if self.run is None:
            return
        try:
            self.run.log({k: v for k, v in d.items() if v != "" and k.split("/")[-1] not in ("step", "split", "time")}, step=step)
        except Exception as e:
            print(f"WARNING: W&B log failed ({e})")

    def curves(self, v, step, n_pred):
        """The whole 1..n_pred error curve of the model against the baselines, as one chart per validation."""
        if self.run is None:
            return
        try:
            xs = [0.2 * k for k in range(1, n_pred + 1)]
            names = ["prior", "post", "empty", "copy", "warp"]
            chart = self.wandb.plot.line_series(xs=xs, ys=[v[k].tolist() for k in names], keys=names,
                                                title=f"val MSE per horizon (step {step})", xname="horizon (s)")
            self.run.log({"val/mse_curve": chart}, step=step)
        except Exception as e:
            print(f"WARNING: W&B curve failed ({e})")

    def image(self, path, step):
        if self.run is None:
            return
        try:
            self.run.log({"val/samples": self.wandb.Image(path, caption="truth / 3 prior samples / prior sigma")},
                         step=step)
        except Exception as e:
            print(f"WARNING: W&B image failed ({e})")

    def finish(self, best):
        if self.run is not None:
            self.run.summary["best_val_mse_prior"] = best
            self.run.finish()


def main():
    a = args_()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    out_dir = os.path.join(a.runs_dir, a.run)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out_dir, "args.json"), "w"), indent=1)

    train_loader, val_loader, n_tr, n_va, stats = loaders(a, cfg)
    print(f"device {dev} | train windows {n_tr} | val windows {n_va} | n_pred {a.n_pred} | prior_sg {a.prior_sg}")

    wm_cfg = WMConfig(encoder=a.encoder, n_cond=cfg.data.n_cond, z_ch=a.z_ch, dropout=a.dropout,
                      p_zdrop=a.p_zdrop, beta=a.beta, prior_sg=a.prior_sg)
    model = WorldModel(wm_cfg).to(dev)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in enc_ids]},
                             {"params": enc_params, "lr": a.lr * a.enc_lr_mult}],
                            lr=a.lr, weight_decay=a.weight_decay)
    pc = param_counts(model)
    print("params (M): " + ", ".join(f"{k} {v / 1e6:.2f}" for k, v in pc.items()))
    json.dump(pc, open(os.path.join(out_dir, "params.json"), "w"), indent=1)
    wb = WandB(a, out_dir, {"params": pc, "n_train_windows": n_tr, "n_val_windows": n_va, "wm_cfg": vars(wm_cfg)})

    step, best = 0, float("inf")
    ckpt = os.path.join(out_dir, "last.pt")
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, best = s["step"], s["best"]
        torch.manual_seed(a.seed + step)                              # do not replay the same z/dropout noise
        print(f"resumed from step {step}")
    elif a.init_encoder:
        assert a.encoder == "vit", "MAE weights only fit the vit encoder"
        model.encoder.load_state_dict(torch.load(a.init_encoder, map_location=dev))
        print(f"encoder initialised from {a.init_encoder} (encoder lr x{a.enc_lr_mult})")

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "wm_cfg": vars(wm_cfg), "stats": stats}, path + ".tmp")
        os.replace(path + ".tmp", path)                              # never leave a half-written checkpoint

    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
    if new_log:
        log.writeheader()

    use_amp = dev == "cuda"
    keys = ("loss", "mse", "mse_warm", "kl", "prior_std")
    last_ckpt, t0 = time.time(), time.time()
    run = {k: 0.0 for k in keys}; run_n = 0
    model.train()
    while step < a.steps:
        for b in train_loader:
            b = to(b, dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                loss, logs = model.loss(b, a.n_pred)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {logs}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            for k in keys:
                run[k] += logs[k]
            run_n += 1

            if step % 100 == 0:
                m = {k: run[k] / run_n for k in keys}
                sps = (time.time() - t0) / run_n
                gu = gpu_util()
                print(f"step {step} | loss {m['loss']:.5f} mse {m['mse']:.5f} warm {m['mse_warm']:.5f} "
                      f"kl {m['kl']:.1f} prior_std {m['prior_std']:.3f} | {sps:.3f} s/step gpu {gu}%")
                log.writerow({"step": step, "split": "train", **m, "s_per_step": sps, "gpu_util": gu,
                              "time": time.time()})
                log_f.flush()
                wb.log({f"train/{k}": x for k, x in m.items()} | {"train/s_per_step": sps, "train/gpu_util": gu},
                       step)
                run = {k: 0.0 for k in keys}; run_n = 0; t0 = time.time()

            if step % a.val_every == 0 or step >= a.steps:
                v = validate(model, val_loader, a, dev, cfg, stats)
                row = {"step": step, "split": "val", "mse": v["prior"].mean().item(), "mse_warm": v["mse_warm"],
                       "kl": v["kl"].mean().item(), "prior_std": v["std"].mean().item(),
                       "act_sens": v["act_sens"], "gap_mean": v["gap"].mean().item(), "time": time.time()}
                print(f"  VAL (step {step})        " + "  ".join(f"{k * 0.2:.0f}s" .rjust(9) for k in REPORT_STEPS))
                for name in ("prior", "post", "empty", "copy", "warp", "mass", "kl", "std"):
                    vals = [v[name][min(k, len(v[name])) - 1].item() for k in REPORT_STEPS]
                    for k, x in zip(REPORT_STEPS, vals):
                        row[f"{name}@{k}"] = x
                    fmt = "{:9.1f}" if name == "kl" else "{:9.5f}" if name not in ("mass", "std") else "{:9.3f}"
                    print(f"    {name:6s}               " + "  ".join(fmt.format(x) for x in vals))
                print(f"    act_sens {v['act_sens']:.5f} | warm-up mse {v['mse_warm']:.5f} | "
                      f"gap(prior-post) mean {v['gap'].mean().item():.5f}")
                log.writerow(row)
                log_f.flush()
                sample_path = os.path.join(out_dir, f"samples_{step:06d}.png")
                val_loader.generator.manual_seed(VAL_SEED)      # same sample scene every validation
                save_samples(model, next(iter(val_loader)), a, dev, cfg.data.n_cond, sample_path)
                wb.log({f"val/{k}": x for k, x in row.items()}, step)
                wb.curves(v, step, a.n_pred)
                wb.image(sample_path, step)
                if v["prior"].mean().item() < best:
                    best = v["prior"].mean().item()
                    save(os.path.join(out_dir, "best.pt"))

            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                save(ckpt); last_ckpt = time.time()
            if step >= a.steps:
                break
    save(ckpt)
    wb.finish(best)
    print(f"done at step {step}, best val mse (prior) {best:.5f}")


if __name__ == "__main__":
    main()
