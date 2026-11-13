"""Action-invariance loss ("fix B"): road users that cannot react to the ego must not depend on the ego's action.

Why. The model learns p(future | past, ego actions) from logs in which the human chose the actions because of the
other road users (the lead brakes -> the human brakes). So the ego's action is evidence about them, and the model
makes the lead copy the ego ("lead mirroring": gain ~0.5 under brake, negative under go; the planner's crashes).
A planner needs p(future | past, do(actions)). For a road user that cannot react to the ego (a lead, a parked car, a
receding car) that is p(its future | past): it must not depend on the action at all, except through the ego's own
motion (the ego-centric frame moves with the ego).

The loss. For a training window, roll the model out twice for H steps from the same warm-up state, with the
human's actions a_h and with a counterfactual a' (another acceleration profile, the same route), z = the prior mean
in both (the leak lives in the deterministic action pathway; learned and fixed priors mirror the same). Warp the a'
prediction into the a_h ego frame with the known ego poses (W), and penalise
    sum M * (W(pred(a')) - pred(a_h))^2 / sum M
where M = the region around road users that cannot react (dilated, mostly along the ego's forward axis, where a
mirrored car is displaced), minus pixels near any other road user, inside the area both frames see. Under the
constraint "no action dependence there", the MSE-optimal prediction for those cars is p(car | past) = p(car | do(a)).
Followers and road users that may interact (yielding ring cars...) are NOT in M: whatever the model learned about
them is left alone (followers really react; the ring cars are the proposal's question).

Which road users cannot react (from the raw tracks at the window's last warm-up frame t0, over the next H steps):
    lead-like   ahead of the ego ON the ego's own future path (within --lat m of it, aligned within 45 degrees):
                a car ahead in the ego's lane does not respond to the car behind it (includes the 2nd car ahead
                and queues; works on the curved ring, unlike a straight corridor)
    parked      < 0.5 m/s for the whole H steps and never within --clear m of the ego's path
    receding    moving, its distance to the ego never shrinks by more than 0.2 m per step, and it never comes
                within --clear m of the ego's path
Everything else present in a frame (followers, crossing or merging ring cars, new arrivals) is "other". A
follower (behind the ego on its past path, travelling its way) is "other" even if its distance never shrinks.
The classification uses the human's ego path; under a' the ego is somewhere else on the same path, which a lead,
a parked or a receding car still cannot notice within 2 s.

Counterfactual actions (torch): an acceleration profile per sample, either a constant from `consts` (m/s^2) or the
human acceleration sequence of another window in the batch (human-like profiles, so the model cannot keep leaking
only for "human-looking" actions); speeds re-integrated with the floor at 0 (and the acceleration recomputed from
them, so the model never sees "braking while stopped"); curvature taken from the human's route BY DISTANCE (as the
planner does: the ego stays on its road).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .kinematics import step_torch, wrap_angle
from .wm_eval import warp_previous

GREEN = 1


# ---- classification (numpy, used by scripts/build_inv_masks.py) ----------------------------------------------------
def path_polyline(xy, heading0, length=60.0, ds=0.5):
    """The ego's future path from its logged positions xy (K, 2) (first = now), resampled every ds m up to `length` m
    of arc; extended straight along the last direction if the logged path is shorter (a stopped ego still has a lane
    ahead). -> points (N, 2), headings (N,), arc length (N,)."""
    xy = np.asarray(xy, dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-3])                     # drop repeated points (standing still)
    xy = xy[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if len(xy) >= 2:
        d = xy[-1] - xy[-2]
        last_dir = d / max(np.linalg.norm(d), 1e-9)
    else:
        last_dir = np.array([np.cos(heading0), np.sin(heading0)])
    if s[-1] < length:                                              # straight extension
        xy = np.vstack([xy, xy[-1] + (length - s[-1]) * last_dir])
        s = np.append(s, length)
    grid = np.arange(0.0, length + 1e-9, ds)
    pts = np.stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])], 1)
    d = np.gradient(pts, axis=0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    return pts, hd, grid


def classify_others(path, ego0, others0, futures, lat=2.5, clear=8.0, tol=0.2, ego_len=4.5, past=None):
    """Which road users cannot react to the ego.

    path      (pts (N, 2), headings (N,), arc (N,)) from path_polyline
    past      the same for the ego's PAST path, walked backwards (path_polyline of the reversed past positions):
              a road user on it, travelling the ego's way, is a follower -> always "other" (it reacts)
    ego0      (x, y) of the ego now
    others0   dict trackId -> (x, y, heading) now
    futures   dict trackId -> (xy (K, 2), speed (K,), ego_xy (K, 2)) over now .. now + H (the steps where the road
              user is present; ego_xy = the logged ego position at the same steps)
    -> dict trackId -> "lead" | "parked" | "receding" | "other"
    """
    pts, hd, arc = path
    out = {}
    for tid, (x, y, h) in others0.items():
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        j = int(d.argmin())
        dh = abs(wrap_angle(h - hd[j]))
        if d[j] < lat and arc[j] > ego_len / 2 and dh < np.pi / 4:
            out[tid] = "lead"
            continue
        if past is not None:                                        # a follower: behind on the ego's own path
            pp, ph, pa = past
            dp = np.hypot(pp[:, 0] - x, pp[:, 1] - y)
            jp = int(dp.argmin())
            if dp[jp] < lat and pa[jp] > ego_len / 2 and abs(wrap_angle(h - ph[jp] - np.pi)) < np.pi / 4:
                out[tid] = "other"
                continue
        fxy, fv, exy = futures[tid]
        # distance of the road user's positions to the ego's path (now and future path) and to the ego itself
        dpath = np.min(np.hypot(fxy[:, None, 0] - pts[None, :, 0], fxy[:, None, 1] - pts[None, :, 1]))
        dpath = min(dpath, float(np.min(np.hypot(fxy[:, 0] - ego0[0], fxy[:, 1] - ego0[1]))))
        if dpath <= clear:
            out[tid] = "other"
        elif fv.max() < 0.5:
            out[tid] = "parked"
        else:
            dist = np.hypot(fxy[:, 0] - exy[:, 0], fxy[:, 1] - exy[:, 1])
            out[tid] = "receding" if (np.diff(dist) >= -tol).all() else "other"
    return out


# ---- counterfactual actions and poses (torch) ----------------------------------------------------------------------
def _raw(actions, stats):
    m = torch.tensor(stats["action_mean"], device=actions.device, dtype=torch.float32)
    s = torch.tensor(stats["action_std"], device=actions.device, dtype=torch.float32)
    return actions.float() * s + m


def _norm(raw, stats):
    m = torch.tensor(stats["action_mean"], device=raw.device, dtype=torch.float32)
    s = torch.tensor(stats["action_std"], device=raw.device, dtype=torch.float32)
    return (raw - m) / s


def arc_lengths(v, dt):
    """v (B, T+1) speeds -> cumulative distance (B, T+1) with the midpoint rule (as kinematics.step_*)."""
    return torch.cat([torch.zeros_like(v[:, :1]), torch.cumsum(0.5 * (v[:, 1:] + v[:, :-1]) * dt, 1)], 1)


def counterfactual_actions(batch, stats, n, H, dt, consts=(-3.0, -1.5, 0.0, 1.5), human_frac=0.5, accel=None,
                           generator=None):
    """Counterfactual inputs for the H rollout steps after the warm-up.

    batch: images, actions (B, L-1, 2) normalised, speed (B, L, 1) normalised, ego_state (B, L, 4) raw.
    accel: optional (B, H) raw accelerations to use (tests); otherwise drawn as described in the module docstring.
    Returns dict: actions, speed (normalised, full window length; only steps n-1 .. n+H-2 changed),
                  raw (acc, curv) for the human and the counterfactual, each (B, H) -> for the poses.
    """
    act, st = batch["actions"], batch["ego_state"].float()
    B, dev = act.shape[0], act.device
    raw = _raw(act, stats)                                          # (B, L-1, 2)
    acc_h, curv_h = raw[:, n - 1:, 0], raw[:, n - 1:, 1]            # the whole future (B, T)
    T = acc_h.shape[1]
    v0 = st[:, n - 1, 3]
    if accel is None:
        g = generator
        pick_h = torch.rand(B, generator=g, device="cpu").to(dev) < human_frac
        shift = int(torch.randint(1, max(B, 2), (1,), generator=g).item()) if B > 1 else 0
        a_hum = torch.roll(acc_h[:, :H], shift, 0)
        c = torch.tensor(consts, device=dev, dtype=torch.float32)
        a_con = c[torch.randint(0, len(consts), (B,), generator=g).to(dev)][:, None].expand(B, H)
        accel = torch.where(pick_h[:, None], a_hum, a_con)
    accel = accel.float()
    # speeds with the floor at 0, then the acceleration that is actually applied
    v = [v0]
    for i in range(H):
        v.append(torch.clamp(v[-1] + accel[:, i] * dt, min=0.0))
    v = torch.stack(v, 1)                                           # (B, H + 1)
    a_eff = (v[:, 1:] - v[:, :-1]) / dt
    # curvature from the human's route by distance: step i of the counterfactual uses the human curvature of the
    # segment of the route where it starts
    v_h = st[:, n - 1:, 3]                                          # (B, T + 1) logged speeds
    s_h = arc_lengths(v_h, dt)                                      # (B, T + 1)
    s_c = arc_lengths(v, dt)[:, :-1]                                # (B, H) arc length at the start of each step
    j = torch.searchsorted(s_h[:, 1:].contiguous(), s_c.contiguous(), right=True).clamp(max=T - 1)
    curv_c = torch.gather(curv_h, 1, j)
    raw_c = raw.clone()
    raw_c[:, n - 1:n - 1 + H, 0] = a_eff
    raw_c[:, n - 1:n - 1 + H, 1] = curv_c
    spd = batch["speed"].clone().float()
    spd[:, n:n + H, 0] = (v[:, 1:] - stats["speed_mean"]) / stats["speed_std"]
    return {"actions": _norm(raw_c, stats).to(act.dtype), "speed": spd.to(batch["speed"].dtype),
            "human": (acc_h[:, :H], curv_h[:, :H]), "cf": (a_eff, curv_c)}


def integrate_poses(state0, acc, curv, dt):
    """state0 (B, 4) raw x, y, heading, speed; acc, curv (B, H) -> poses (B, H, 3) after each step."""
    s, out = state0.double(), []
    for i in range(acc.shape[1]):
        s = step_torch(s, torch.stack([acc[:, i].double(), curv[:, i].double()], -1), dt)
        out.append(s[:, :3])
    return torch.stack(out, 1)


def warp_to(green, pose_src, pose_tgt, rc):
    """green (B, H, 1, 64, 64) seen from pose_src (B, H, 3) -> the same scene seen from pose_tgt (B, H, 3), and the
    share of each target pixel that the source frame covers (B, H, 1, 64, 64)."""
    B, H = green.shape[:2]
    src = green.float().flatten(0, 1)                               # (BH, 1, 64, 64)
    ps, pt = pose_src.flatten(0, 1), pose_tgt.flatten(0, 1)[:, None]
    w = warp_previous(src, ps, pt, rc)[:, 0]                        # (BH, 1, 64, 64)
    valid = warp_previous(torch.ones_like(src), ps, pt, rc)[:, 0]
    return w.unflatten(0, (B, H)), valid.unflatten(0, (B, H))


def region(nr, re, fwd=8, side=2, near=2):
    """nr, re (B, H, 64, 64) bool footprints (cannot react / others) -> M (B, H, 1, 64, 64) float: nr dilated by
    `fwd` px along the ego's forward axis (rows) and `side` px across, minus pixels within `near` px of others."""
    B, H = nr.shape[:2]
    x = nr.float().flatten(0, 1)[:, None]
    d = F.max_pool2d(x, (2 * fwd + 1, 2 * side + 1), 1, (fwd, side))
    o = F.max_pool2d(re.float().flatten(0, 1)[:, None], 2 * near + 1, 1, near)
    return (d * (1 - o)).unflatten(0, (B, H))


# ---- the loss --------------------------------------------------------------------------------------------------------
def _rollout(model, h, actions, speeds, n, H, prev):
    greens = []
    for k in range(H):
        i = n - 1 + k
        if model.cfg.deterministic:
            z = model.zero_z(h.shape[0], h.device)
        else:
            z = model.prior_params(h)[0].detach()                   # the prior mean; the prior head is not trained here
        h, g = model.rollout_step(h, z, actions[:, i], speeds[:, i], None, prev)
        greens.append(g)
        prev = g
    return torch.stack(greens, 1)                                   # (B, H, 1, 64, 64)


def invariance_loss(model, batch, stats, rc, dt, H=10, consts=(-3.0, -1.5, 0.0, 1.5), human_frac=0.5,
                    fwd=8, side=2, near=2, accel=None, generator=None):
    """-> (loss, logs). batch needs inv_nr, inv_re (B, >= H, 64, 64) bool from the mask cache.
    Call inside the same fixed_dropout_masks() block as the main loss: both rollouts then use the same dropout masks
    (otherwise their difference would include dropout noise)."""
    if model.cfg.reencode:
        raise ValueError("the invariance loss does not support --reencode (no road/ego frames under a')")
    n = model.cfg.n_cond
    imgs, act, spd = batch["images"], batch["actions"], batch["speed"]
    e = model.encode(imgs[:, :n])
    h0, _ = model.warmup(e, act, spd)
    prev = imgs[:, n - 1, GREEN:GREEN + 1]
    cf = counterfactual_actions(batch, stats, n, H, dt, consts, human_frac, accel, generator)
    g_h = _rollout(model, h0, act, spd, n, H, prev)
    g_c = _rollout(model, h0, cf["actions"], cf["speed"], n, H, prev)
    st0 = batch["ego_state"][:, n - 1].float()
    pose_h = integrate_poses(st0, *cf["human"], dt)
    pose_c = integrate_poses(st0, *cf["cf"], dt)
    g_cw, valid = warp_to(g_c, pose_c, pose_h, rc)
    M = region(batch["inv_nr"][:, :H], batch["inv_re"][:, :H], fwd, side, near) * (valid > 0.99).float()
    area = M.sum()
    loss = (M * (g_cw - g_h.float()) ** 2).sum() / area.clamp_min(1.0)
    shift = (pose_c[..., :2] - pose_h[..., :2]).norm(dim=-1)[:, -1].mean()
    return loss, {"inv": loss.item(), "inv_area": (area / M[:, :, 0].numel()).item(), "inv_shift": shift.item()}
