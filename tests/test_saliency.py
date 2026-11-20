"""scripts/saliency.py: the action map is ~0 on road users that don't react (real replay frames) and lights up for
road users glued to the ego; smoke runs on an old-style and a DINO checkpoint."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import fake_round
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.dataset import compute_stats
from roundabout_wm.models.dino_wm import hidden_size
from roundabout_wm.models.world_model import WMConfig, WorldModel
from saliency import action_map

_S = {}


def _setup():
    if not _S:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        stats = compute_stats(cache, [0], cfg)
        cks = []
        for name, wc in (("old", WMConfig(n_cond=cfg.data.n_cond)),
                         ("dino", WMConfig(n_cond=cfg.data.n_cond, arch="dino", dino="stub",
                                           hidden=hidden_size("stub", 4)))):
            torch.manual_seed(0)
            ck = os.path.join(root, "runs", name, "last.pt")
            os.makedirs(os.path.dirname(ck))
            torch.save({"model": WorldModel(wc).state_dict(), "wm_cfg": vars(wc), "stats": stats}, ck)
            cks.append(ck)
        _S.update(root=root, cfg=cfg, raw=raw, cache=cache, stats=stats, cks=cks)
    return _S


def _replay(d=0.5):
    S = _setup()
    env = ReplayEnv(S["cache"], S["raw"], [0], S["cfg"], S["stats"], horizon=10)
    env.reset(fake_round.ego_window(env))
    v0 = float(env.item["ego_state"][S["cfg"].data.n_cond - 1, 3])
    t = np.arange(11) * S["cfg"].data.dt
    speeds = np.stack([np.full(11, v0), v0 + d * t, np.maximum(v0 - d * t, 0)])
    out = env.open_loop(speeds)
    g = torch.as_tensor(out["green"], dtype=torch.float32) / 255.0          # (3, T, 64, 64) in [0, 1]
    poses = torch.as_tensor(out["poses"][:, 1:, :3], dtype=torch.float32)  # (3, T, 3)
    return S, g, poses


def test_nobody_reacts_in_replay_so_the_action_map_is_dark():
    S, g, poses = _replay()
    A = action_map(g[1], g[2], poses[1], poses[2], poses[0], S["cfg"].render, 0.5)
    glued = action_map(g[0], g[0], poses[1], poses[2], poses[0], S["cfg"].render, 0.5)  # frames ignore the ego
    car = g[0] > 0.3
    late = slice(7, 10)                       # 1.6-2.0 s: the plans have separated by 1.3-2 m
    real, fake = A[late].abs()[car[late]].mean().item(), glued[late].abs()[car[late]].mean().item()
    assert fake > 0.25 and real < 0.2 * fake, (real, fake)      # measured: 0.03 (interpolation) vs 0.31


def test_smoke_old_and_dino_checkpoints():
    S = _setup()
    env = ReplayEnv(S["cache"], S["raw"], [0], S["cfg"], S["stats"], horizon=10)
    w = fake_round.ego_window(env)
    out = os.path.join(S["root"], "sal")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "saliency.py"), "--cache", S["cache"],
                        "--data_dir", S["raw"], "--split", "val", "--ckpts", *S["cks"], "--windows", str(w),
                        "--dirs", "2", "--out_dir", out], capture_output=True, text=True, cwd=S["root"],
                       env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert os.path.exists(os.path.join(out, f"saliency_{w}.png"))
    d = pd.read_csv(os.path.join(out, "saliency.csv"))
    assert set(d.run) == {"old", "dino"} and d.act_on_empty.notna().all()
