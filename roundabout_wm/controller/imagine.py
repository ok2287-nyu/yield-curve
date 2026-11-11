"""Batched imagined rollouts of many acceleration plans from one model state.

    paths = ego_paths(state0, route_s, route_k, accels, cfg)          # kinematics only (numpy)
    out = imagine(model, h0, paths, stats, cfg, n_samples=4, n_masks=4, seed=0)
    out["green"]   (M, S, P, T, 64, 64)  predicted other road users, frames 1..T after the start
    out["speed"]   (P, T + 1)            ego speed at the start of each step and after the last

Steering is not controlled: every plan follows the human's route, curvature looked up by distance travelled (the
same rule as sim.ImaginedDrive), so plans differ only in acceleration.

Two kinds of randomness, kept apart:
  S prior samples (aleatoric): z = mu_p(h) + sigma_p(h) * eps, eps drawn per (sample, plan) and SHARED by all M
    dropout masks, so differences between masks are not differences in sampled futures.
  M dropout masks (epistemic): with M > 1 the model runs in train mode inside fixed_dropout_masks(); mask m is the
    same for every sample and plan (one ensemble member scores every plan), and held over the whole rollout.
    M = 1: eval mode, no dropout.
"""
from __future__ import annotations

import numpy as np
import torch

from ..kinematics import step_np
from ..models.world_model import StickyDropout2d


def route_curvature(route_s, route_k, s):
    """Curvature of the human's route at distance s (same lookup as sim.ImaginedDrive.step)."""
    j = np.searchsorted(route_s, np.asarray(s) + 1e-6, side="right") - 1
    return np.asarray(route_k)[np.clip(j, 0, len(route_k) - 1)]


def ego_paths(state0, route_s, route_k, accels, cfg):
    """state0 (4,) x, y, heading, speed; accels (P, T) m/s^2. Follows the route by distance.
    -> dict state (P, T + 1, 4), acc (P, T) clipped, curv (P, T) used."""
    d = cfg.data
    acc = np.clip(np.asarray(accels, dtype=np.float64), -d.accel_clip, d.accel_clip)
    P, T = acc.shape
    st = np.repeat(np.asarray(state0, dtype=np.float64)[None], P, 0)
    s = np.zeros(P)
    states, curvs = [st], []
    for t in range(T):
        k = np.clip(route_curvature(route_s, route_k, s), -d.curvature_clip, d.curvature_clip)
        k = np.where(st[:, 3] >= d.v_min_curvature, k, 0.0)
        nxt = step_np(st, np.stack([acc[:, t], k], -1), d.dt)
        s = s + 0.5 * (st[:, 3] + nxt[:, 3]) * d.dt
        st = nxt
        states.append(st); curvs.append(k)
    return {"state": np.stack(states, 1), "acc": acc, "curv": np.stack(curvs, 1)}


def _share_masks(model, M, rep):
    """Inside fixed_dropout_masks(): sample M masks per dropout module (one dummy step with batch M), then repeat each
    mask `rep` times, so batch row m * rep + j uses mask m."""
    C = model.cfg.hidden
    dev = next(model.parameters()).device
    z = model.zero_z(M, dev)
    model.rollout_step(torch.zeros(M, C, 8, 8, device=dev), z, torch.zeros(M, 2, device=dev),
                       torch.zeros(M, 1, device=dev), prev=torch.zeros(M, 1, 64, 64, device=dev))
    for mod in model.modules():
        if isinstance(mod, StickyDropout2d) and mod.mask is not None:
            mod.mask = mod.mask.repeat_interleave(rep, 0)


def imagine(model, h0, paths, stats, cfg, n_samples=4, n_masks=1, seed=0, prev0=None, grad=False):
    """h0 (1, C, 8, 8): model state at the start (after the real warm-up). paths: ego_paths(...).
    prev0 (1, 1, 64, 64): the current green frame (residual decoder only). -> dict green, speed, acc, curv."""
    if model.cfg.reencode:
        raise NotImplementedError("reencode models need rendered frames inside the rollout")
    if model.cfg.z_seq:
        raise NotImplementedError("per-sequence latent models are not supported by the planner")
    dev = h0.device
    acc, curv, state = paths["acc"], paths["curv"], paths["state"]
    P, T = acc.shape
    S, M = (1 if model.cfg.deterministic else n_samples), n_masks
    SP = S * P
    B = M * SP
    a_mean, a_std = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    act_n = (np.stack([acc, curv], -1) - a_mean) / a_std                                 # (P, T, 2)
    spd_n = (state[:, :, 3] - stats["speed_mean"]) / stats["speed_std"]                  # (P, T + 1)
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    act_b = to(act_n)[None].expand(S, P, T, 2).reshape(SP, T, 2).repeat(M, 1, 1)         # row = m*SP + s*P + p
    spd_b = to(spd_n)[None].expand(S, P, T + 1).reshape(SP, T + 1).repeat(M, 1)
    g = torch.Generator(device=dev).manual_seed(seed)
    was_training = model.training
    ctx = torch.enable_grad() if grad else torch.no_grad()
    greens = []
    with ctx:
        if M > 1:
            model.train()
            cm = model.fixed_dropout_masks()
            cm.__enter__()
            _share_masks(model, M, SP)
        try:
            h = h0.expand(B, *h0.shape[1:]).contiguous()
            prev = (prev0 if prev0 is not None else torch.zeros(1, 1, 64, 64, device=dev)).expand(B, 1, 64, 64)
            for t in range(T):
                if model.cfg.deterministic:
                    z = model.zero_z(B, dev)
                else:
                    mu, lv = model.prior_params(h)
                    eps = torch.randn((SP,) + tuple(mu.shape[1:]), generator=g, device=dev).repeat(
                        (M,) + (1,) * (mu.dim() - 1))
                    z = mu + (0.5 * lv).exp() * eps
                h, gr = model.rollout_step(h, z, act_b[:, t], spd_b[:, t:t + 1], prev=prev)
                prev = gr
                greens.append(gr[:, 0].float())
        finally:
            if M > 1:
                cm.__exit__(None, None, None)
            model.train(was_training)
    green = torch.stack(greens, 1).view(M, S, P, T, 64, 64)
    return {"green": green, "speed": state[:, :, 3], "acc": acc, "curv": curv, "state": state}


def route_from_item(item, stats, cfg):
    """From a dataset window: the ego state at the last warm-up frame, the human's route from there (curvature
    against distance, as sim.ImaginedDrive.reset), and the human's raw accelerations from there.
    -> state0 (4,), route_s (L+1,), route_k (L,), logged_acc (L,)."""
    n, dt = cfg.data.n_cond, cfg.data.dt
    raw = item["actions"].numpy() * np.asarray(stats["action_std"]) + np.asarray(stats["action_mean"])
    st = item["ego_state"].numpy().astype(np.float64)
    v = st[n - 1:, 3]
    seg = 0.5 * (v[:-1] + v[1:]) * dt
    return st[n - 1].copy(), np.concatenate([[0.0], np.cumsum(seg)]), raw[n - 1:, 1], raw[n - 1:, 0]


@torch.no_grad()
def warm_states(model, items, cfg, device):
    """Warm-up hidden states (B, C, 8, 8) and the last real green frames (B, 1, 64, 64) for a list of windows."""
    n = cfg.data.n_cond
    imgs = torch.stack([it["images"][:n] for it in items]).to(device)
    act = torch.stack([it["actions"] for it in items]).to(device)
    spd = torch.stack([it["speed"] for it in items]).to(device)
    h, _ = model.warmup(model.encode(imgs), act, spd)
    return h, imgs[:, n - 1, 1:2]
