"""The action-invariance loss (roundabout_wm/invariance.py, scripts/build_inv_masks.py, train_wm --inv_weight)."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import fake_round
from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows, compute_stats
from roundabout_wm.invariance import (classify_others, counterfactual_actions, integrate_poses, invariance_loss,
                                      path_polyline, region, warp_to)
from roundabout_wm.models.world_model import WMConfig, WorldModel
from roundabout_wm.render import render_frame

CFG = Config()
_S = {}


def _setup():
    if not _S:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        out = os.path.join(root, "inv")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_inv_masks.py"), "--cache", cache,
                            "--data_dir", raw, "--out", out, "--workers", "1"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        stats = compute_stats(cache, [0], cfg)
        _S.update(root=root, raw=raw, cache=cache, inv=out, stats=stats)
    return _S


def _rows(bits):
    b = np.unpackbits(bits).reshape(64, 64)
    r = np.flatnonzero(b.any(1))
    return (int(r.min()), int(r.max())) if len(r) else None


# ---- classification -------------------------------------------------------------------------------------------
def test_classify_on_a_straight_road():
    # ego at the origin driving +x at 6 m/s; positions over 2 s (11 steps of 0.2 s)
    t = np.arange(11) * 0.2
    ego = np.stack([6 * t, 0 * t], 1)
    path = path_polyline(np.stack([6 * np.arange(30) * 0.2, np.zeros(30)], 1), 0.0)
    past = path_polyline(np.stack([-6 * np.arange(10) * 0.2, np.zeros(10)], 1), np.pi, length=40.0)
    tracks = {
        1: (20 + 6 * t, 0 * t, 0.0, 6.0),          # lead, same speed
        2: (-12 + 6 * t, 0 * t, 0.0, 6.0),         # follower at a constant gap: NOT receding
        3: (5 + 0 * t, 30 + 0 * t, 0.0, 0.0),      # parked 30 m to the side
        4: (0 * t, 15 + 8 * t, np.pi / 2, 8.0),    # driving away on a side road
        5: (40 + 0 * t, 20 - 8 * t, -np.pi / 2, 8.0),   # approaching the ego's path from the side (crossing)
        6: (60 + 0 * t, 0 * t, 0.0, 0.0),          # standing in the lane 60 m ahead (end of the path): lead
    }
    others0 = {k: (x[0], y[0], h) for k, (x, y, h, v) in tracks.items()}
    fut = {k: (np.stack([x, y], 1), np.full(11, v), ego) for k, (x, y, h, v) in tracks.items()}
    got = classify_others(path, ego[0], others0, fut, past=past)
    assert got == {1: "lead", 2: "other", 3: "parked", 4: "receding", 5: "other", 6: "lead"}


def test_masks_of_the_synthetic_scene():
    S = _setup()
    st = np.load(os.path.join(S["inv"], "rec_00_starts.npy"))
    nr = np.load(os.path.join(S["inv"], "rec_00_nr.npy"))
    re = np.load(os.path.join(S["inv"], "rec_00_re.npy"))
    assert nr.shape[1:] == (10, 512)
    i = int(np.flatnonzero(st == 0)[0])         # ego 1: leader 20 m ahead (row ~16), follower 12 m behind (~59)
    lo, hi = _rows(nr[i, 0])
    assert 10 <= lo and hi <= 22
    lo, hi = _rows(re[i, 0])
    assert lo >= 52
    i = int(np.flatnonzero(st == 120)[0])       # ego 3 (the follower): both cars ahead are leads
    assert _rows(re[i, 0]) is None and _rows(nr[i, 0])[1] <= 31


def test_dataset_reads_the_masks():
    S = _setup()
    ds = RounDWindows(S["cache"], [0], CFG, S["stats"], inv_dir=S["inv"], inv_steps=10)
    it = ds[0]
    assert it["inv_nr"].shape == (10, 64, 64) and it["inv_nr"].dtype == torch.bool
    assert it["inv_nr"].any() and it["inv_re"].any()
    plain = RounDWindows(S["cache"], [0], CFG, S["stats"])
    assert "inv_nr" not in plain[0]


# ---- counterfactual actions, poses, warp -------------------------------------------------------------------------
def _batch(ds, idx):
    items = [ds[i] for i in idx]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def test_human_accel_reproduces_the_human_inputs():
    S = _setup()
    ds = RounDWindows(S["cache"], [0], CFG, S["stats"], inv_dir=S["inv"])
    b = _batch(ds, [0, 3])
    n, H = CFG.data.n_cond, 10
    acc_h = b["actions"][:, n - 1:n - 1 + H, 0] * S["stats"]["action_std"][0] + S["stats"]["action_mean"][0]
    cf = counterfactual_actions(b, S["stats"], n, H, CFG.data.dt, accel=acc_h)
    assert torch.allclose(cf["actions"][:, :n - 1 + H], b["actions"][:, :n - 1 + H], atol=1e-4)
    assert torch.allclose(cf["speed"][:, :n + H], b["speed"][:, :n + H], atol=1e-4)


def test_warp_moves_a_logged_car_to_the_right_place():
    """Render the leader from where the ego is under 'brake', warp it into the human frame: it must land on the
    leader rendered from the human pose (the pose integration, the curvature by distance and the warp agree)."""
    S = _setup()
    ds = RounDWindows(S["cache"], [0], CFG, S["stats"])
    b = _batch(ds, [0])                             # ego 1, window start row 0: frame k <-> t = k * 0.2 s
    n, H, dt = CFG.data.n_cond, 10, CFG.data.dt
    cf = counterfactual_actions(b, S["stats"], n, H, dt, accel=torch.full((1, H), -3.0))
    st0 = b["ego_state"][:, n - 1].float()
    ph, pc = integrate_poses(st0, *cf["human"], dt), integrate_poses(st0, *cf["cf"], dt)
    assert (pc[0, :, 0] < ph[0, :, 0]).all()        # braking: behind the human

    def leader(pose, k):
        t = (n + k) * dt
        ego = dict(x=float(pose[0]), y=float(pose[1]), heading=float(pose[2]), length=4.5, width=1.8)
        o = {kk: np.array([v]) for kk, v in dict(x=30 + 6 * t, y=-50.0, heading=0.0, length=4.5, width=1.8).items()}
        return render_frame(ego, o, None, CFG.render)[1].astype(np.float32) / 255

    g_c = torch.tensor(np.stack([leader(pc[0, k], k) for k in range(H)]))[None, :, None]
    g_h = torch.tensor(np.stack([leader(ph[0, k], k) for k in range(H)]))[None, :, None]
    w, valid = warp_to(g_c, pc, ph, CFG.render)
    err = (w - g_h).abs().sum((2, 3, 4))[0]
    moved = (g_c - g_h).abs().sum((2, 3, 4))[0]
    assert moved[-1] > 20                           # the plans really differ by the end (~6 m = 8 px)
    assert (err[-1] < 0.25 * moved[-1]) and (err < 0.25 * moved.clamp_min(4) + 2).all()
    assert valid[0, -1, 0, 43, 31] == 1


def test_region_grows_forward_and_avoids_others():
    nr = torch.zeros(1, 1, 64, 64, dtype=torch.bool); nr[0, 0, 20, 30] = True
    re = torch.zeros_like(nr); re[0, 0, 30, 30] = True
    M = region(nr, re, fwd=8, side=2, near=2)[0, 0, 0]
    assert M[12, 30] == 1 and M[20, 32] == 1 and M[20, 33] == 0      # 8 px along rows, 2 across
    assert M[28, 30] == 0                                             # within 2 px of another road user


# ---- the loss ------------------------------------------------------------------------------------------------------
def _model(det=False):
    torch.manual_seed(0)
    return WorldModel(WMConfig(n_cond=CFG.data.n_cond, deterministic=det))


def test_loss_is_zero_for_the_human_actions_and_positive_otherwise():
    S = _setup()
    ds = RounDWindows(S["cache"], [0], CFG, S["stats"], inv_dir=S["inv"])
    b = _batch(ds, [0, 3, 6])
    n, H = CFG.data.n_cond, 10
    acc_h = b["actions"][:, n - 1:n - 1 + H, 0] * S["stats"]["action_std"][0] + S["stats"]["action_mean"][0]
    for det in (True, False):
        m = _model(det).eval()
        l0, logs = invariance_loss(m, b, S["stats"], CFG.render, CFG.data.dt, H=H, accel=acc_h)
        assert l0.item() < 1e-6 and logs["inv_area"] > 0
        l1, _ = invariance_loss(m, b, S["stats"], CFG.render, CFG.data.dt, H=H, accel=torch.full((3, H), -3.0))
        assert l1.item() > 1e-6
        l1.backward()
        assert m.A[0].weight.grad.abs().sum() > 0                    # the action pathway gets a gradient


def test_same_dropout_masks_inside_the_fixed_block():
    """In training mode, both rollouts must use the same dropout masks: with the human's actions the loss is 0."""
    S = _setup()
    ds = RounDWindows(S["cache"], [0], CFG, S["stats"], inv_dir=S["inv"])
    b = _batch(ds, [0, 3])
    n, H = CFG.data.n_cond, 10
    acc_h = b["actions"][:, n - 1:n - 1 + H, 0] * S["stats"]["action_std"][0] + S["stats"]["action_mean"][0]
    m = _model().train()
    with m.fixed_dropout_masks():
        l0, _ = invariance_loss(m, b, S["stats"], CFG.render, CFG.data.dt, H=H, accel=acc_h)
    assert l0.item() < 1e-6


def test_train_wm_smoke_with_the_invariance_loss():
    S = _setup()
    run = os.path.join(S["root"], "runs")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_wm.py"), "--cache", S["cache"],
                        "--run", "inv", "--runs_dir", run, "--steps", "2", "--batch", "4", "--workers", "0",
                        "--val_every", "1000", "--val_batches", "1", "--inv_weight", "0.1",
                        "--inv_masks", S["inv"]], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert os.path.exists(os.path.join(run, "inv", "last.pt"))
    assert "inv" in pd.read_csv(os.path.join(run, "inv", "log.csv")).columns
