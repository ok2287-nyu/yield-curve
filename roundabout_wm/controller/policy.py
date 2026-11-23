"""Learned policy trained through the frozen world model (controller B; PPUU / MPUR-style, policy design Nov 22-23).

    pi(h, speed, curvature ahead) -> acceleration in [A_LO, A_HI] m/s^2
    h: the world model's hidden state (C x 8 x 8): warmed on the 10 real frames at the start of a rollout and in
       replay; imagined along a training rollout. Steering follows the human's route by distance, as everywhere.

Training (scripts/train_policy.py), per batch of train-split windows:
    h0 = warm-up on the real frames; then for T steps: a = pi(h, v, curvature ahead); v' = max(v + a dt, 0);
    z = mu_p(h) + sigma_p(h) * eps, ONE prior sample per step (PPUU: "approximated by a single sample from the
    prior"); h, green = model.rollout_step(h, z, (a, curvature), v). Gradients flow through the frozen model.
    Cost = cost.plan_costs' terms (collision, proximity, progress, comfort) + lam * U, with U PPUU's uncertainty
    cost: the variance over K dropout masks (same h, z, action) of the predicted green frame, summed over pixels,
    normalised per step by its mean / std under the human's own actions on train windows, hinged at 0.

policy_rollout() with fixed `accels` instead of a policy reproduces controller/imagine.py exactly (tested), so
planner and policy see the same imagined world.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost import CostConfig, reference_speed, step_risk
from .imagine import route_curvature

A_LO, A_HI = -3.0, 1.5
CURV_AHEAD = (0.0, 5.0, 10.0, 20.0)          # m ahead of the ego along the route
CURV_SCALE = 20.0                            # curvature x this ~ O(1) at roundabout radii (~20 m)


class PolicyNet(nn.Module):
    """h (B, C, 8, 8), speed (B,) normalised, curvature ahead (B, 4) 1/m -> acceleration (B,) in [a_lo, a_hi]."""

    def __init__(self, hidden=256, width=64, mlp=256, a_lo=A_LO, a_hi=A_HI):
        super().__init__()
        self.a_lo, self.a_hi = a_lo, a_hi
        self.conv = nn.Sequential(nn.Conv2d(hidden, width, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(width, width, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.mlp = nn.Sequential(nn.Linear(width * 16 + 1 + len(CURV_AHEAD), mlp), nn.LeakyReLU(0.2),
                                 nn.Linear(mlp, mlp), nn.LeakyReLU(0.2), nn.Linear(mlp, 1))
        nn.init.zeros_(self.mlp[-1].weight)                  # start as "coast" (a = 0) everywhere
        mid, half = (a_hi + a_lo) / 2, (a_hi - a_lo) / 2
        nn.init.constant_(self.mlp[-1].bias, float(np.arctanh(np.clip((0.0 - mid) / half, -0.99, 0.99))))

    def forward(self, h, v_n, curv):
        x = torch.cat([self.conv(h.float()).flatten(1), v_n.float()[:, None], curv.float() * CURV_SCALE], 1)
        mid, half = (self.a_hi + self.a_lo) / 2, (self.a_hi - self.a_lo) / 2
        return mid + half * torch.tanh(self.mlp(x)[:, 0])


def routes_from_batch(batch, stats, cfg):
    """Batched imagine.route_from_item: -> v0 (B,), route_s (B, L+1), route_k (B, L), acc_prev (B,) raw m/s^2,
    logged accelerations from the last warm-up frame (B, L)."""
    n, dt = cfg.data.n_cond, cfg.data.dt
    raw = batch["actions"].cpu().numpy() * np.asarray(stats["action_std"]) + np.asarray(stats["action_mean"])
    st = batch["ego_state"].cpu().numpy().astype(np.float64)
    v = st[:, n - 1:, 3]
    seg = 0.5 * (v[:, :-1] + v[:, 1:]) * dt
    route_s = np.concatenate([np.zeros((len(v), 1)), np.cumsum(seg, 1)], 1)
    return v[:, 0], route_s, raw[:, n - 1:, 1], raw[:, n - 2, 0], raw[:, n - 1:, 0]


def curvature_at(route_s, route_k, s, cfg):
    """Per batch element b: route curvature at arc length s[b] (+ offsets), clipped. s (B,) or (B, k)."""
    s = np.asarray(s, dtype=np.float64)
    out = np.stack([route_curvature(route_s[b], route_k[b], s[b]) for b in range(len(s))])
    return np.clip(out, -cfg.data.curvature_clip, cfg.data.curvature_clip)


def policy_rollout(model, h0, prev0, v0, route_s, route_k, stats, cfg, T, policy=None, accels=None, eps=None,
                   n_masks=0):
    """Roll the world model forward T steps from h0 with the policy's (or given) accelerations.
    h0 (B, C, 8, 8); prev0 (B, 1, 64, 64) the last real green frame; v0 (B,) m/s; route_s / route_k per element.
    eps: list of T noise tensors (B, z_ch, 8, 8) (default: drawn), or "mean" for z = mu_p.
    n_masks K > 0: also the PPUU uncertainty U_t = sum over pixels of the variance over K dropout masks of the
    green frame of step t (same h, z, action). -> dict green (B, T, 64, 64), v (B, T + 1), acc (B, T), curv (B, T),
    unc (B, T) or None. The model is run in eval mode (no dropout) except for the K uncertainty passes."""
    if model.cfg.reencode or model.cfg.z_seq or model.cfg.z_global:
        raise NotImplementedError("policy rollouts need a spatial per-step latent and no re-encoding")
    d = cfg.data
    dev = h0.device
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=dev)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=dev)
    sn = lambda v: (v - stats["speed_mean"]) / stats["speed_std"]
    B = h0.shape[0]
    v = torch.as_tensor(v0, dtype=torch.float32, device=dev)
    s = np.zeros(B)
    h, prev = h0, prev0
    was = model.training
    model.eval()
    greens, vs, accs, curvs, uncs = [], [v], [], [], []
    try:
        for t in range(T):
            vd = v.detach().cpu().numpy()
            k = curvature_at(route_s, route_k, s, cfg)
            k = np.where(vd >= d.v_min_curvature, k, 0.0)
            k_t = torch.as_tensor(k, dtype=torch.float32, device=dev)
            if policy is not None:
                ahead = curvature_at(route_s, route_k, s[:, None] + np.asarray(CURV_AHEAD), cfg)
                a = policy(h, sn(v), torch.as_tensor(ahead, dtype=torch.float32, device=dev))
            else:
                a = torch.as_tensor(accels[:, t], dtype=torch.float32, device=dev)
            a = a.clamp(-d.accel_clip, d.accel_clip)
            v2 = torch.clamp(v + a * d.dt, min=0.0)
            act_n = (torch.stack([a, k_t], -1) - a_mean) / a_std
            spd_n = sn(v)[:, None]
            mu, lv = model.prior_params(h)
            if isinstance(eps, str) and eps == "mean":
                z = mu
            else:
                e = eps[t] if eps is not None else torch.randn_like(mu)
                z = mu + (0.5 * lv).exp() * e
            if n_masks:
                model.train()
                gs = torch.stack([model.rollout_step(h, z, act_n, spd_n, prev=prev)[1] for _ in range(n_masks)])
                model.eval()
                uncs.append(gs.float().var(0, unbiased=True).sum((-1, -2, -3)))
            h, g = model.rollout_step(h, z, act_n, spd_n, prev=prev)
            s = s + 0.5 * (vd + v2.detach().cpu().numpy()) * d.dt
            v, prev = v2, g
            greens.append(g[:, 0].float()); vs.append(v); accs.append(a); curvs.append(k_t)
    finally:
        model.train(was)
    return {"green": torch.stack(greens, 1), "v": torch.stack(vs, 1), "acc": torch.stack(accs, 1),
            "curv": torch.stack(curvs, 1), "unc": torch.stack(uncs, 1) if uncs else None}


def rollout_cost(out, cfg, cc: CostConfig, acc_prev, unc_norm=None, lam=0.0):
    """Per-element costs (B,) of a policy rollout, with plan_costs' terms; plus lam * hinged normalised U.
    unc_norm: (mean (T,), std (T,)) of U under the human's actions (calibrate_uncertainty)."""
    dt = cfg.data.dt
    v_after = out["v"][:, 1:]
    col, prox = step_risk(out["green"], v_after, cfg.render, cc)                      # (B, T) each
    risk = (cc.w_col * col + cc.w_prox * prox).sum(-1)
    v_ref = torch.as_tensor(reference_speed(out["curv"].detach().cpu().numpy(), cc), dtype=torch.float32,
                            device=v_after.device)
    prog = (torch.clamp(1 - v_after / v_ref, min=0) ** 2).sum(-1)
    a = out["acc"]
    a0 = torch.as_tensor(acc_prev, dtype=torch.float32, device=a.device)[:, None]
    da = torch.diff(torch.cat([a0, a], 1), dim=1) / dt
    comfort = cc.w_acc * (a ** 2).sum(-1) + cc.w_jerk * (da ** 2).sum(-1)
    unc = torch.zeros_like(risk)
    if out["unc"] is not None and unc_norm is not None:
        mu, sd = (torch.as_tensor(x, dtype=torch.float32, device=risk.device) for x in unc_norm)
        unc = F.relu((out["unc"] - mu) / sd).mean(-1)
    total = risk + cc.w_prog * prog + comfort + lam * unc
    return {"total": total, "risk": risk, "progress": prog, "comfort": comfort, "unc": unc,
            "collision": col.sum(-1), "proximity": prox.sum(-1)}


class LearnedPolicy:
    """The trained PolicyNet as a ReplayEnv policy: the same real-frame buffers and warm-up as the model planner,
    then one forward pass per step."""

    def __init__(self, model, policy, stats, cfg, name="policy", device=None):
        from .planner import ModelPlanner
        self._buf = ModelPlanner(model, stats, cfg, samples=1, masks=1, name=name, device=device)
        self.pi, self.stats, self.cfg, self.name = policy.eval(), stats, cfg, name
        self.dev = self._buf.dev
        self.accs = []

    def reset(self, env):
        self._buf.reset(env)

    @torch.no_grad()
    def act(self, env, obs):
        if obs["t"] > 0:
            self._buf._observe(obs)
        h, _ = self._buf.warm()
        v = float(obs["state"][3])
        rs, rk = self._buf.route_s - float(obs["s"]), self._buf.route_k
        ahead = curvature_at([rs], [rk], np.asarray([CURV_AHEAD]), self.cfg)
        v_n = torch.tensor([(v - self.stats["speed_mean"]) / self.stats["speed_std"]], device=self.dev)
        a = float(self.pi(h, v_n, torch.as_tensor(ahead, dtype=torch.float32, device=self.dev))[0])
        self.accs.append(a)
        return a

    def episode_summary(self):
        out = {"pol_acc_mean": float(np.mean(self.accs)) if self.accs else np.nan,
               "pol_at_limit": float(np.mean([(x <= A_LO + 0.05) or (x >= A_HI - 0.05) for x in self.accs]))
               if self.accs else np.nan}
        self.accs = []
        return out
