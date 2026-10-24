"""PyTorch dataset over the cached recordings.

Each item is one window of n_cond + n_pred frames from one ego track (n_pred is set per dataset):
    images      float (L, 3, H, W) in [0, 1]       L = n_cond + n_pred
    speed       float (L, 1)                       ego speed (normalised if stats given)
    actions     float (L-1, 2)                     action k moves frame k -> k+1 (normalised if stats given)
    ego_state   float (L, 4)                       x, y, heading, speed in world units (bookkeeping only)

Typical world-model use: condition on images[:n_cond], speed[:n_cond];
predict images[n_cond:] given actions[n_cond-1:].
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config


def window_starts(cache_rec_dir: str, window_len: int, stride: int) -> np.ndarray:
    """Start rows of every window of `window_len` frames that fits inside one ego track.

    Computed from tracks.csv, so the rollout length can change without rebuilding the cache.
    """
    tr = pd.read_csv(os.path.join(cache_rec_dir, "tracks.csv"))
    starts = [np.arange(s, s + n - window_len + 1, stride) for s, n in zip(tr["start"], tr["length"])
              if n >= window_len]
    return np.concatenate(starts).astype(np.int64) if starts else np.zeros(0, np.int64)


class RounDWindows(Dataset):
    """n_pred overrides cfg.data.n_pred (e.g. 30 or 50 for longer rollouts); no cache rebuild needed."""

    def __init__(self, cache_dir: str, rec_ids: list[int], cfg: Config, stats: dict | None = None,
                 n_pred: int | None = None):
        self.cfg = cfg
        self.n_pred = n_pred or cfg.data.n_pred
        self.L = cfg.data.n_cond + self.n_pred
        self.recs, index = [], []
        for r in rec_ids:
            d = os.path.join(cache_dir, f"rec_{r:02d}")
            rec = {"frames": np.load(os.path.join(d, "frames.npy"), mmap_mode="r"),
                   "states": np.load(os.path.join(d, "states.npy")),
                   "actions": np.load(os.path.join(d, "actions.npy"))}
            w = window_starts(d, self.L, cfg.data.window_stride)
            index.append(np.stack([np.full(len(w), len(self.recs)), w], 1))
            self.recs.append(rec)
        self.index = np.concatenate(index) if index else np.zeros((0, 2), np.int64)
        self.stats = stats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, s = self.index[i]
        rec, L = self.recs[k], self.L
        images = torch.from_numpy(np.array(rec["frames"][s:s + L])).float().div_(255.0)  # copy out of the memmap
        state = torch.from_numpy(rec["states"][s:s + L].copy())
        actions = torch.from_numpy(rec["actions"][s:s + L - 1].copy())
        speed = state[:, 3:4].clone()
        if self.stats is not None:
            speed = (speed - self.stats["speed_mean"]) / self.stats["speed_std"]
            actions = (actions - torch.tensor(self.stats["action_mean"])) / torch.tensor(self.stats["action_std"])
        return {"images": images, "speed": speed, "actions": actions, "ego_state": state}


def compute_stats(cache_dir: str, rec_ids: list[int], cfg: Config) -> dict:
    """Mean/std of speed and actions over the rows used by training windows (train split only!)."""
    speeds, acts = [], []
    L = cfg.data.cache_min_track_len      # frozen: keeps stats.json identical across teammates' caches
    for r in rec_ids:
        d = os.path.join(cache_dir, f"rec_{r:02d}")
        st, ac = (np.load(os.path.join(d, f)) for f in ("states.npy", "actions.npy"))
        w = window_starts(d, L, cfg.data.window_stride)
        rows = np.unique((w[:, None] + np.arange(L - 1)[None]).ravel())
        speeds.append(st[rows, 3]); acts.append(ac[rows])
    speeds, acts = np.concatenate(speeds), np.concatenate(acts)
    return {"speed_mean": float(speeds.mean()), "speed_std": float(speeds.std() + 1e-6),
            "action_mean": acts.mean(0).tolist(), "action_std": (acts.std(0) + 1e-6).tolist()}


def save_stats(stats: dict, path: str):
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)


def load_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
