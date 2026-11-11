"""Cost of an imagined (or real) future for the ego, computed on the green channel around the ego's fixed pixel.

Per step t (frame t+1 after the action):
    collision   sum(green * F) / sum(F), F = the ego's footprint (soft, as rendered)              weight w_col
    proximity   max over pixels of green * Prox_v, Prox_v = 1 alongside the ego, falling linearly to 0 at
                prox_base + prox_time * v metres ahead of its front and at prox_half_width metres to the side;
                nothing behind the ego (followers are not the ego's fault)                         weight w_prox
    progress    max(0, 1 - v / v_ref)^2,  v_ref = min(v_max, sqrt(a_lat / |curvature|))           weight w_prog
    comfort     acc^2 and ((acc_t - acc_{t-1}) / dt)^2                                            w_acc, w_jerk
Per plan: the risk (collision + proximity terms) and progress / comfort are summed over the T steps and the risk is
averaged over prior samples (aleatoric outcomes are planned for, not penalised as spread). Epistemic: the std of
the summed risk over dropout masks (same prior noise in every mask), penalised only above epi_thresh
(calibrate: about the 90th percentile under the human's own actions), weight w_epi.
Weights are starting values (Fable 5.2's plan, Nov 11); calibrate before trusting them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..render import _fill_rects


@dataclass
class CostConfig:
    w_col: float = 10.0
    w_prox: float = 1.0
    w_prog: float = 0.2
    w_acc: float = 0.02
    w_jerk: float = 0.01
    w_epi: float = 0.5
    epi_thresh: float = 0.0
    prox_base: float = 4.0         # m ahead of the ego's front at standstill
    prox_time: float = 0.5         # s: the zone grows by this many seconds of travel
    prox_half_width: float = 2.5   # m to each side of the ego's centre line
    v_max: float = 10.0            # m/s, progress reference on straight road
    a_lat: float = 2.0             # m/s^2, comfortable lateral acceleration for the reference speed in turns
    ego_length: float = 4.5
    ego_width: float = 1.8

    def to_dict(self):
        return asdict(self)


def ego_footprint(rc, cc: CostConfig):
    """(64, 64) soft footprint of the ego at its fixed pixel, pointing up."""
    m = rc.m_per_px
    hl, hw = cc.ego_length / 2 / m, cc.ego_width / 2 / m
    c, r = rc.ego_col, rc.ego_row
    corners = np.array([[[c - hw, r - hl], [c + hw, r - hl], [c + hw, r + hl], [c - hw, r + hl]]])
    return torch.from_numpy(_fill_rects(corners, rc).astype(np.float32) / 255.0)


def proximity_mask(speed, rc, cc: CostConfig):
    """speed (...,) m/s -> (..., 64, 64) proximity weights."""
    speed = torch.as_tensor(speed, dtype=torch.float32)
    idx = torch.arange(rc.size, dtype=torch.float32)
    fwd = (rc.ego_row - idx)[:, None].expand(rc.size, rc.size) * rc.m_per_px          # metres ahead of the centre
    lat = (rc.ego_col - idx)[None, :].expand(rc.size, rc.size).abs() * rc.m_per_px    # metres to the side
    front, back, hw = cc.ego_length / 2, -cc.ego_length / 2, cc.ego_width / 2
    reach = (cc.prox_base + cc.prox_time * speed)[..., None, None]                      # (..., 1, 1)
    w_long = torch.clamp(1 - (fwd - front) / reach, 0, 1)
    w_long = torch.where(fwd <= front, torch.ones_like(w_long), w_long)
    w_long = torch.where(fwd < back, torch.zeros_like(w_long), w_long)
    w_lat = torch.clamp(1 - (lat - hw) / (cc.prox_half_width - hw), 0, 1)
    return w_long * w_lat


def reference_speed(curv, cc: CostConfig):
    k = np.abs(np.asarray(curv, dtype=np.float64))
    return np.minimum(cc.v_max, np.sqrt(cc.a_lat / np.maximum(k, 1e-6)))


def step_risk(green, speed_after, rc, cc: CostConfig):
    """green (..., T, 64, 64); speed_after (P, T) or broadcastable to green's (..., T). -> (collision, proximity),
    each (..., T)."""
    F = ego_footprint(rc, cc).to(green.device)
    col = (green * F).sum((-1, -2)) / F.sum()
    prox_m = proximity_mask(speed_after, rc, cc).to(green.device)
    prox = (green * prox_m).flatten(-2).amax(-1)
    return col, prox


def plan_costs(green, speed, acc, curv, rc, cc: CostConfig, acc_prev: float = 0.0, dt: float = 0.2):
    """green (M, S, P, T, 64, 64); speed (P, T + 1); acc, curv (P, T). -> dict of per-plan tensors (P,):
    total, risk (mean over masks and samples), epistemic (std over masks), progress, comfort; plus per-step
    collision and proximity (M, S, P, T)."""
    dev = green.device
    v_after = torch.as_tensor(speed[:, 1:], dtype=torch.float32, device=dev)          # (P, T)
    col, prox = step_risk(green, v_after, rc, cc)                                      # (M, S, P, T)
    risk_ms = (cc.w_col * col + cc.w_prox * prox).sum(-1)                              # (M, S, P)
    risk = risk_ms.mean((0, 1))
    epi = risk_ms.mean(1).std(0, unbiased=True) if risk_ms.shape[0] > 1 else torch.zeros_like(risk)
    v_ref = torch.as_tensor(reference_speed(curv, cc), dtype=torch.float32, device=dev)
    prog = (torch.clamp(1 - v_after / v_ref, min=0) ** 2).sum(-1)
    a = torch.as_tensor(acc, dtype=torch.float32, device=dev)
    da = torch.diff(torch.cat([torch.full_like(a[:, :1], acc_prev), a], 1), dim=1) / dt
    comfort = cc.w_acc * (a ** 2).sum(-1) + cc.w_jerk * (da ** 2).sum(-1)
    total = risk + cc.w_epi * torch.clamp(epi - cc.epi_thresh, min=0) + cc.w_prog * prog + comfort
    return {"total": total, "risk": risk, "epistemic": epi, "progress": prog, "comfort": comfort,
            "collision": col, "proximity": prox}
