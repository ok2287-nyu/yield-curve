"""MAE-pretrain the vit encoder on single frames from the TRAIN recordings (the vit_mae run's first stage).

    python scripts/pretrain_mae.py --cache $C --run mae
    -> runs/mae/encoder.pt     encoder state_dict only, loaded by: train_wm.py --encoder vit --init_encoder runs/mae/encoder.pt

Rerunning the same command resumes from runs/<run>/last.pt (checkpoint every --ckpt_minutes). Every --log_every steps
the masked-patch loss is logged separately for patches with a car (green > 0) and without: most patches are empty road,
so a low overall loss can hide that cars are not learned.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from roundabout_wm.models.mae import MAE, patchify


class Frames(Dataset):
    """Every cached frame of the given recordings, one 3x64x64 float image in [0, 1] per item (memory-mapped)."""

    def __init__(self, cache, recs):
        self.paths = [os.path.join(cache, f"rec_{r:02d}", "frames.npy") for r in recs]
        self.sizes = [np.load(p, mmap_mode="r").shape[0] for p in self.paths]
        self.ends = np.cumsum(self.sizes)
        self.maps = {}

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.ends, i, side="right"))
        if k not in self.maps:
            self.maps[k] = np.load(self.paths[k], mmap_mode="r")
        j = i - (int(self.ends[k - 1]) if k else 0)
        return torch.from_numpy(np.array(self.maps[k][j])).float() / 255.0


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--run", default="mae")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--keep", type=int, default=16, help="visible patches out of 64 (16 = 25 percent)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--viz_every", type=int, default=2000)
    p.add_argument("--ckpt_minutes", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_recs", type=int, default=None, help="debug: use only the first N train recordings")
    return p.parse_args()


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    return 0.5 * a.lr * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))


@torch.no_grad()
def save_viz(model, x, path):
    """Rows (4 images): input with masked patches greyed, MAE reconstruction (visible patches pasted back), truth."""
    model.eval()
    idx, masked = model.random_keep(x.shape[0], x.device)
    _, pred, _, _ = model(x, idx)
    tgt = patchify(x)
    mu, sd = tgt.mean(-1, keepdim=True), tgt.std(-1, keepdim=True) + 1e-3
    rec = (pred * sd + mu).clamp(0, 1)                                         # undo the per-patch normalisation
    rec = torch.where(masked[..., None], rec, tgt)
    inp = torch.where(masked[..., None], torch.full_like(tgt, 0.5), tgt)

    def img(t):                                                                # (B, 64, 192) -> (B, 3, 64, 64)
        B = t.shape[0]
        return t.reshape(B, 8, 8, 3, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 64, 64)

    rows = [torch.cat(list(img(t)), 2) for t in (inp, rec, tgt)]
    grid = torch.cat(rows, 1)
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) \
        .resize((grid.shape[2] * 3, grid.shape[1] * 3), Image.NEAREST).save(path)
    model.train()


def main():
    a = args_()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(a.runs_dir, a.run)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out_dir, "args.json"), "w"), indent=1)

    recs = json.load(open(os.path.join(a.cache, "split.json")))["train"]           # train recordings only
    if a.max_recs:
        recs = recs[:a.max_recs]
    ds = Frames(a.cache, recs)
    g = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, generator=g, num_workers=a.workers, drop_last=True,
                        pin_memory=True, persistent_workers=a.workers > 0)
    viz_x = torch.stack([ds[i] for i in np.linspace(0, len(ds) - 1, 8).astype(int)]).to(dev)
    print(f"device {dev} | train recordings {recs[0]}..{recs[-1]} | frames {len(ds)} | keep {a.keep}/64")

    model = MAE(keep=a.keep).to(dev)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params (M): encoder {n_enc / 1e6:.2f}, MAE total {n_all / 1e6:.2f}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)

    step = 0
    ckpt = os.path.join(out_dir, "last.pt")
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        torch.manual_seed(a.seed + step)
        print(f"resumed from step {step}")

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def save_encoder():
        torch.save(model.encoder.state_dict(), os.path.join(out_dir, "encoder.pt"))

    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    fields = ["step", "loss", "loss_car", "loss_empty", "car_frac", "lr", "s_per_step", "time"]
    log = csv.DictWriter(log_f, fieldnames=fields)
    if new_log:
        log.writeheader()

    use_amp = dev == "cuda"
    last_ckpt, t0 = time.time(), time.time()
    acc = {"loss": 0.0, "car": 0.0, "empty": 0.0, "car_n": 0.0, "empty_n": 0.0}
    n_acc = 0
    model.train()
    while step < a.steps:
        for x in loader:
            x = x.to(dev, non_blocking=True)
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, a)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                loss, _, per_patch, masked = model(x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1

            with torch.no_grad():
                car = model.car_patches(x) & masked
                emp = ~model.car_patches(x) & masked
                pp = per_patch.float()
                acc["loss"] += loss.item()
                acc["car"] += (pp * car).sum().item(); acc["car_n"] += car.sum().item()
                acc["empty"] += (pp * emp).sum().item(); acc["empty_n"] += emp.sum().item()
                n_acc += 1

            if step % a.log_every == 0:
                m = {"step": step, "loss": acc["loss"] / n_acc,
                     "loss_car": acc["car"] / max(acc["car_n"], 1), "loss_empty": acc["empty"] / max(acc["empty_n"], 1),
                     "car_frac": acc["car_n"] / max(acc["car_n"] + acc["empty_n"], 1),
                     "lr": lr_at(step, a), "s_per_step": (time.time() - t0) / n_acc, "time": time.time()}
                print(f"step {step} | loss {m['loss']:.4f} car-patch {m['loss_car']:.4f} empty-patch "
                      f"{m['loss_empty']:.4f} (car share {m['car_frac']:.3f}) | {m['s_per_step']:.3f} s/step")
                log.writerow(m); log_f.flush()
                acc = {k: 0.0 for k in acc}; n_acc = 0; t0 = time.time()
            if step % a.viz_every == 0 or step >= a.steps:
                save_viz(model, viz_x, os.path.join(out_dir, f"recon_{step:06d}.png"))
                save_encoder()
            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                save(ckpt); last_ckpt = time.time()
            if step >= a.steps:
                break
    save(ckpt)
    save_encoder()
    print(f"done at step {step}; encoder weights in {os.path.join(out_dir, 'encoder.pt')}")


if __name__ == "__main__":
    main()
