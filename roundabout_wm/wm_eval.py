"""Evaluation helpers for the world model: baselines and counterfactual action inputs (torch).

warp_previous   the green channel of the last observed frame, re-projected into each future ego frame using only
                the logged ego poses ("the others stand still in the world, only I move"). Same pixel convention as
                render.py: pixel centres at integer coordinates, row = ego_row - forward/m, col = ego_col - left/m.
swap_actions    give every sample in the batch another sample's future actions, and recompute the future speeds
                from them with the kinematic model, so (action, speed) stays consistent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import RenderConfig


def warp_previous(green_src, pose_src, pose_tgt, rc: RenderConfig):
    """green_src (B, 1, H, W) at pose_src (B, 3) = (x, y, heading); pose_tgt (B, T, 3) -> (B, T, 1, H, W)."""
    B, T = pose_tgt.shape[:2]
    H = W = rc.size
    dev = green_src.device
    rows, cols = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64),
                                torch.arange(W, device=dev, dtype=torch.float64), indexing="ij")
    fwd_t = (rc.ego_row - rows) * rc.m_per_px                                   # target-frame offsets (H, W)
    left_t = (rc.ego_col - cols) * rc.m_per_px
    ps, pt = pose_src.double(), pose_tgt.double()
    ct, st = torch.cos(pt[..., 2])[..., None, None], torch.sin(pt[..., 2])[..., None, None]   # (B, T, 1, 1)
    # world point minus the SOURCE ego position (subtract first: world coordinates are ~100 m)
    dx = (pt[..., 0] - ps[:, None, 0])[..., None, None] + fwd_t * ct - left_t * st
    dy = (pt[..., 1] - ps[:, None, 1])[..., None, None] + fwd_t * st + left_t * ct
    cs, ss = torch.cos(ps[:, 2])[:, None, None, None], torch.sin(ps[:, 2])[:, None, None, None]
    fwd_s = dx * cs + dy * ss
    left_s = -dx * ss + dy * cs
    col_s = rc.ego_col - left_s / rc.m_per_px
    row_s = rc.ego_row - fwd_s / rc.m_per_px
    grid = torch.stack([col_s / (W - 1) * 2 - 1, row_s / (H - 1) * 2 - 1], -1).float()   # (B, T, H, W, 2)
    src = green_src.float()[:, None].expand(B, T, 1, H, W).flatten(0, 1)
    out = F.grid_sample(src, grid.flatten(0, 1), mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.unflatten(0, (B, T))


def _reintegrate_speeds(act, ego_state, n_cond, stats, dt):
    """Future speeds implied by (normalised) actions: v[i+1] = max(v[i] + acc[i]*dt, 0) from the sample's own
    v[n_cond-1] (kinematics.step_torch), returned normalised (B, L, 1). Warm-up speeds are the logged ones."""
    a_mean = torch.tensor(stats["action_mean"], device=act.device)
    a_std = torch.tensor(stats["action_std"], device=act.device)
    acc = act[..., 0] * a_std[0] + a_mean[0]                                    # raw m/s^2, (B, L-1)
    v = ego_state[:, :, 3].clone()                                              # raw m/s, (B, L)
    for i in range(n_cond - 1, v.shape[1] - 1):
        v[:, i + 1] = torch.clamp(v[:, i] + acc[:, i] * dt, min=0.0)
    return ((v - stats["speed_mean"]) / stats["speed_std"])[..., None]


def swap_actions(actions, speeds, ego_state, n_cond, stats, dt, shift=1):
    """Counterfactual inputs: rollout actions act[n_cond-1:] taken from sample (b - shift) of the batch.

    Warm-up actions/speeds are untouched; future speeds are re-integrated from the swapped accelerations.
    Returns (actions, speeds), normalised.
    """
    act = actions.clone()
    act[:, n_cond - 1:] = torch.roll(actions[:, n_cond - 1:], shift, 0)
    return act, _reintegrate_speeds(act, ego_state, n_cond, stats, dt).to(speeds.dtype)


def counterfactual_accel(actions, ego_state, n_cond, stats, dt, accel):
    """Same scene, but every rollout step uses a constant acceleration `accel` (m/s^2); curvature as logged.
    Returns (actions, speeds), normalised. (The speed floor at 0 means 'brake' ends in a stop.)"""
    a_mean = torch.tensor(stats["action_mean"], device=actions.device)
    a_std = torch.tensor(stats["action_std"], device=actions.device)
    act = actions.clone()
    act[:, n_cond - 1:, 0] = (accel - a_mean[0]) / a_std[0]
    return act, _reintegrate_speeds(act, ego_state, n_cond, stats, dt).to(actions.dtype)
