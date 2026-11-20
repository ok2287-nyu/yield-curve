"""Who does the model think the ego controls? Sensitivity maps of the predicted frames to the ego's ACTION and to the
LATENT, over a 2 s rollout. (CPU or GPU; images contain rounD: keep them on HPC / your machine.)

    python scripts/saliency.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_anchor/last.pt runs/ab_inv2_01/last.pt --csv runs/ab_anchor/action_gain.csv --top 4

Windows: --windows (dataset indices, as in action_gain.csv), or --csv: the --top windows with the lowest lead gain
under --plan at 2 s (the strongest mirroring), or with --order spread from lowest to highest.

How (finite differences, all rollouts of a window in one batch with the SAME latent noise, so only the perturbation
differs):
  action map  A_k = [ W(g_k | acc + d) - W(g_k | acc - d) ] / 2d,   d = --delta m/s^2 added to EVERY step's
              acceleration (the human's log plan otherwise). W = warp into the camera of the unperturbed (human)
              rollout at the same step, so the camera's own motion cancels: a road user that does not react to the
              ego has A = 0. Orange = the model puts MORE car there when the ego accelerates harder, blue = less.
              A car the model pushes forward shows orange in front, blue behind. Units: frame intensity per m/s^2.
  z map       S_k = rms over R random directions v of [ g_k(eps + e v) - g_k(eps) ] / e, eps the latent noise of all
              steps (z = mu + sigma * eps), e = --eps_step: which pixels the latent controls (per prior std).
  z@1 map     the same with only step 1's noise perturbed: how long one latent "decision" lasts in the rollout.
Deterministic models get only the action map.

Printed per window and model, per step: action response on cars (mean |A| where the human rollout has a car) and on
the empty road (should be ~0); z@1 response relative to step 1 (persistence).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from inspect_wm import compose, labelled_grid, to_img
from roundabout_wm.config import Config
from roundabout_wm.controller.imagine import ego_paths, route_from_item, warm_states
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.controller.warp_plan import warp_frames
from roundabout_wm.models.world_model import WMConfig, WorldModel

GREEN = 1


def args_():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--windows", type=int, nargs="*", default=None)
    p.add_argument("--csv", default=None, help="action_gain.csv: pick lead windows by gain")
    p.add_argument("--plan", default="brake")
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--order", choices=["low", "spread"], default="low")
    p.add_argument("--T", type=int, default=10, help="steps (0.2 s each)")
    p.add_argument("--delta", type=float, default=0.25, help="acceleration perturbation, m/s^2")
    p.add_argument("--eps_step", type=float, default=0.3, help="latent perturbation, in prior std")
    p.add_argument("--dirs", type=int, default=8, help="random latent directions")
    p.add_argument("--a_scale", type=float, default=0.3, help="action map: |A| that is drawn at full colour")
    p.add_argument("--z_scale", type=float, default=0.5, help="z maps: response drawn at full white")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/saliency")
    return p.parse_args()


def load(path, dev):
    s = torch.load(path, map_location=dev)
    m = WorldModel(WMConfig(**s["wm_cfg"])).to(dev).eval()
    m.load_state_dict(s["model"])
    return m, s["stats"]


def action_map(g_up, g_dn, pose_up, pose_dn, pose_base, rc, delta):
    """g_up / g_dn (T, 64, 64): frames under acc + delta / acc - delta, seen from pose_up / pose_dn (T, 3). Both warped
    into pose_base's camera; -> (T, 64, 64) d frame / d acc, 0 where either warp leaves its frame (new territory)."""
    up = warp_frames(g_up[None], pose_up, pose_base, rc)[0]
    dn = warp_frames(g_dn[None], pose_dn, pose_base, rc)[0]
    ones = torch.ones_like(g_up)[None]
    ok = (warp_frames(ones, pose_up, pose_base, rc)[0] > 0.99) & (warp_frames(ones, pose_dn, pose_base, rc)[0] > 0.99)
    return (up - dn) / (2 * delta) * ok


def pick_windows(a):
    if a.windows:
        return list(a.windows)
    d = pd.read_csv(a.csv)
    d = d[(d.group == "lead") & (d.plan == a.plan) & (d.t.round(1) == 2.0)].dropna(subset=["gain"])
    d = d.sort_values("gain").drop_duplicates("window")
    if a.order == "spread" and len(d) > a.top:
        d = d.iloc[np.linspace(0, len(d) - 1, a.top).round().astype(int)]
    return d.window.head(a.top).astype(int).tolist()


@torch.no_grad()
def rollouts(model, h0, prev0, acc_n, spd_n, eps):
    """acc_n (B, T, 2), spd_n (B, T + 1) normalised; eps (B, T, *z) the latent noise per row -> greens (B, T, 64, 64).
    Dropout off (eval); deterministic models ignore eps."""
    B, T = acc_n.shape[:2]
    h = h0.expand(B, *h0.shape[1:]).contiguous()
    prev = prev0.expand(B, 1, 64, 64)
    out = []
    for t in range(T):
        if model.cfg.deterministic:
            z = model.zero_z(B, h.device)
        else:
            mu, lv = model.prior_params(h)
            z = mu + (0.5 * lv).exp() * eps[:, t]
        h, g = model.rollout_step(h, z, acc_n[:, t], spd_n[:, t:t + 1], prev=prev)
        prev = g
        out.append(g[:, 0].float())
    return torch.stack(out, 1)


def maps(model, stats, cfg, env, a, dev):
    """-> dict base (T,64,64), A (T,64,64), Z, Z1 (T,64,64) or None, truth (T,64,64), road, ego (T,64,64)."""
    item, n, T = env.item, cfg.data.n_cond, a.T
    h0, prev0 = warm_states(model, [item], cfg, dev)
    state0, route_s, route_k, logged = route_from_item(item, stats, cfg)
    acc = logged[:T]
    accs = np.stack([acc, acc + a.delta, acc - a.delta])
    paths = ego_paths(state0, route_s, route_k, accs, cfg)
    am, asd = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    act = (np.stack([paths["acc"], paths["curv"]], -1) - am) / asd                       # (3, T, 2)
    spd = (paths["state"][:, :, 3] - stats["speed_mean"]) / stats["speed_std"]           # (3, T + 1)
    poses = torch.as_tensor(paths["state"][:, 1:, :3], dtype=torch.float32, device=dev)  # pose after each step

    zshape = tuple(model.zero_z(1, dev).shape[1:])
    g = torch.Generator(device=dev).manual_seed(a.seed)
    eps0 = torch.randn((1, T) + zshape, generator=g, device=dev)
    R = 0 if model.cfg.deterministic else a.dirs
    V = torch.randn((R, T) + zshape, generator=g, device=dev)
    V = V / V.flatten(2).norm(dim=2).clamp_min(1e-8)[..., None, None, None] * np.sqrt(np.prod(zshape))  # unit rms
    V1 = V.clone()
    V1[:, 1:] = 0                                                                         # step 1 only
    eps = torch.cat([eps0.expand(3, *eps0.shape[1:]), eps0 + a.eps_step * V, eps0 + a.eps_step * V1])
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    rows_act = torch.cat([to(act), to(act[:1]).expand(2 * R, -1, -1)])
    rows_spd = torch.cat([to(spd), to(spd[:1]).expand(2 * R, -1)])
    G = rollouts(model, h0, prev0, rows_act, rows_spd, eps)                              # (3 + 2R, T, 64, 64)

    base = G[0]
    A = action_map(G[1], G[2], poses[1], poses[2], poses[0], cfg.render, a.delta)
    Z = Z1 = None
    if R:
        Z = (((G[3:3 + R] - base) / a.eps_step) ** 2).mean(0).sqrt()
        Z1 = (((G[3 + R:] - base) / a.eps_step) ** 2).mean(0).sqrt()
    fut = item["images"][n:n + T]
    return {"base": base.cpu(), "A": A.cpu(), "Z": None if Z is None else Z.cpu(), "Z1": None if Z1 is None else Z1.cpu(),
            "truth": fut[:, GREEN].float(), "road": fut[:, 0:1].float(), "ego": fut[:, 2:3].float()}


def diverging(A, road, scale):
    """(T, 64, 64) signed -> (T, 3, 64, 64): orange = +, blue = -, on a faint road."""
    x = (A / scale).clamp(-1, 1)
    pos, neg = x.clamp(min=0), (-x).clamp(min=0)
    out = torch.zeros(A.shape[0], 3, *A.shape[1:])
    out[:, 0] = 0.25 * road[:, 0] + pos
    out[:, 1] = 0.55 * pos + 0.45 * neg
    out[:, 2] = neg
    return out.clamp(0, 1)


def gray(S, road, scale):
    x = (S / scale).clamp(0, 1)
    return (x[:, None].expand(-1, 3, -1, -1) + 0.2 * road * torch.tensor([1.0, 0, 0])[None, :, None, None]).clamp(0, 1)


def main():
    a = args_()
    cfg = Config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    models = [(os.path.basename(os.path.dirname(os.path.abspath(p))),) + load(p, dev) for p in a.ckpts]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, models[0][2], horizon=a.T)
    cols = list(range(1, a.T, 2))                                       # +0.4, +0.8, ... s
    col_times = [(k + 1) * cfg.data.dt for k in cols]
    rows_csv = []
    for w in pick_windows(a):
        env.reset(w)
        out = None
        rows, labels = [], []
        for name, m, st in models:
            r = maps(m, st, cfg, env, a, dev)
            if out is None:
                out = r
                rows.append(to_img(compose(r["truth"][cols, None], r["road"][cols], r["ego"][cols])))
                labels.append("real | human")
            rows.append(to_img(compose(r["base"][cols, None], r["road"][cols], r["ego"][cols])))
            labels.append(f"{name} | human")
            rows.append(to_img(diverging(r["A"][cols], r["road"][cols], a.a_scale)))
            labels.append(f"{name} d/d acc")
            if r["Z"] is not None:
                rows.append(to_img(gray(r["Z"][cols], r["road"][cols], a.z_scale))); labels.append(f"{name} d/d z")
                rows.append(to_img(gray(r["Z1"][cols], r["road"][cols], a.z_scale))); labels.append(f"{name} d/d z@1")
            car = r["base"] > 0.3
            empty = (r["base"] < 0.05) & (r["truth"] < 0.05)
            for k in range(a.T):
                rows_csv.append(dict(window=w, run=name, t=round((k + 1) * cfg.data.dt, 1),
                                     act_on_cars=float(r["A"][k].abs()[car[k]].mean()) if car[k].any() else np.nan,
                                     act_on_empty=float(r["A"][k].abs()[empty[k]].mean()),
                                     z_on_cars=float(r["Z"][k][car[k]].mean()) if r["Z"] is not None and car[k].any()
                                     else np.nan,
                                     z1_total=float(r["Z1"][k].sum()) if r["Z1"] is not None else np.nan))
        labelled_grid(rows, labels, col_times).save(os.path.join(a.out_dir, f"saliency_{w}.png"))
    d = pd.DataFrame(rows_csv)
    d.to_csv(os.path.join(a.out_dir, "saliency.csv"), index=False)
    d["z1_rel"] = d.z1_total / d.groupby(["window", "run"]).z1_total.transform("first")
    print(f"windows {sorted(d.window.unique().tolist())}; action d = {a.delta} m/s^2, latent step {a.eps_step} prior std")
    print("mean over windows, per step: |d frame / d acc| on cars vs on empty road (per m/s^2); latent response on "
          "cars; step-1 latent influence relative to step 1")
    print(d.groupby(["run", "t"])[["act_on_cars", "act_on_empty", "z_on_cars", "z1_rel"]].mean().round(4)
          .to_string())
    print(f"wrote {d.window.nunique()} sheets to {a.out_dir}/")


if __name__ == "__main__":
    main()
