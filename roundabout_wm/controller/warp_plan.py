"""Predict once, then warp: a planner-side baseline in which no road user can depend on the ego's action.

The model is rolled out only under reference plan(s) that do not use the human's future (coast, or the mix of the
four constant plans). Each candidate plan's future is that reference future seen from where the plan takes the ego:
the reference frames are shifted and rotated with the known ego poses (wm_eval.warp_previous). By construction every
road user then has action gain 1: nobody mirrors the ego, but nobody reacts to it either (like log replay). In
expectation over the reference, this is what the invariance loss aims for on road users that cannot react,
E[car | past], without training. (Sobal et al. 2022 decouple the environment model from the ego the same way.)

Known limits:
    - edge strips: the area a plan sees that the reference frame did not (up to ~8 px at 2 s under brake / go) is
      empty, so cars entering there are missing;
    - followers do not react: under brake they are drawn into the ego (as in replay's "real" with followers), so the
      collision cost of braking includes them;
    - bilinear resampling softens cars slightly.
Reference names: "coast" (the coast plan) or "mix" (brake, gentle, coast, go, averaged). Never the human's own plan,
which would hand the predictor the human's future decisions.
"""
from __future__ import annotations

import torch

from .imagine import imagine
from ..wm_eval import warp_previous

PLAN_NAMES = ("log", "brake", "gentle", "coast", "go")
REFS = {"coast": ("coast",), "mix": ("brake", "gentle", "coast", "go")}


def warp_frames(green, pose_src, pose_tgt, rc):
    """green (B, T, 64, 64) seen from pose_src (T, 3); -> the same frames seen from pose_tgt (T, 3), (B, T, 64, 64)."""
    B, T = green.shape[:2]
    src = green.float().permute(1, 0, 2, 3).reshape(T * B, 1, *green.shape[2:])           # step-major
    ps = pose_src[:, None].expand(T, B, 3).reshape(T * B, 3)
    pt = pose_tgt[:, None, None].expand(T, B, 1, 3).reshape(T * B, 1, 3)
    w = warp_previous(src, ps, pt, rc)[:, 0, 0]                                           # (T*B, 64, 64)
    return w.view(T, B, *w.shape[1:]).permute(1, 0, 2, 3)


def warp_imagine(model, h0, paths, stats, cfg, n_samples=4, n_masks=1, seed=0, prev0=None, ref="coast",
                 z_mode="sample", names=PLAN_NAMES):
    """Same inputs and output as imagine(); green (M, S, P, T, 64, 64) built by warping the reference rollouts."""
    ref_idx = [names.index(r) for r in REFS[ref]]
    sub = {k: paths[k][ref_idx] for k in ("acc", "curv", "state")}
    out = imagine(model, h0, sub, stats, cfg, n_samples, n_masks, seed=seed, prev0=prev0, z_mode=z_mode)
    G = out["green"]                                                                       # (M, S, R, T, 64, 64)
    M, S, R, T = G.shape[:4]
    dev = G.device
    poses = torch.as_tensor(paths["state"][:, 1:T + 1, :3], dtype=torch.float64, device=dev)   # (P, T, 3)
    P = poses.shape[0]
    green = torch.zeros(M, S, P, T, *G.shape[-2:], device=dev)
    for p in range(P):
        acc = 0
        for r, ri in enumerate(ref_idx):
            src = G[:, :, r].reshape(M * S, T, *G.shape[-2:])
            acc = acc + warp_frames(src, poses[ri], poses[p], cfg.render)
        green[:, :, p] = (acc / R).view(M, S, T, *G.shape[-2:])
    return {"green": green, "speed": paths["state"][:, :, 3], "acc": paths["acc"], "curv": paths["curv"],
            "state": paths["state"]}
