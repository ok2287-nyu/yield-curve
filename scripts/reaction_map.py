"""Does the model predict that other drivers react to what the ego does? Plan A vs plan B from the same moment.

    python scripts/reaction_map.py --ckpt runs/cnn_det/best.pt --cache ../round_cache --data_dir ../round_data \
        --scene 0 --a brake --b go --seconds 4 --k 8

In the ego-centric frames, plan B's picture differs from plan A's mostly because the camera moved with the ego. So B's
prediction at each step is re-projected into A's ego frame at that step (exact warp by the known pose difference,
`wm_eval.warp_previous`); what does not cancel is the model's predicted REACTION of the other road users:
    red   = B predicts less traffic there than A           green = B predicts more traffic there than A
Only pixels inside both crops are compared. For a stochastic model both plans are averaged over --k prior samples
(the same k seeds for A and B), so the map is a difference of expected occupancy.

Writes <ckpt dir>/reaction_scene<scene>_<a>_vs_<b>.png with rows:
    A (A's frame) | B (B's frame) | B warped into A's frame | difference (B - A)
(road dim, ego A blue, plan B's ego as a white outline in the last two rows), and prints per 0.4 s the reaction
    score: mean |B - A| over the shared area, and how much traffic B adds / removes.
Caveat (the research question's core problem): the model learned from logs where human egos chose actions after
reading the other drivers, so a predicted "reaction" can be correlation, not causation. Read it together with the lag
(how soon after the actions diverge the difference appears) and only for plausible plans.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image, ImageDraw

from roundabout_wm.sim import ImaginedDrive
from roundabout_wm.wm_eval import warp_previous

NAMED = {"log": None, "brake": -3.0, "go": 1.5, "coast": 0.0}


def plan_value(p):
    return NAMED[p] if p in NAMED else float(p)


@torch.no_grad()
def expected(sim, accel, steps, k, base_seed):
    """Mean over k prior samples (one sample for a deterministic model) of the green prediction, + the ego poses and
    the road at each step (both identical across samples: the ego's motion does not depend on z)."""
    gs = []
    for j in range(k):
        sim.restore("start"); sim.reseed(base_seed + j)
        g, poses, roads = sim.rollout(accel, steps)
        gs.append(g)
    return torch.stack(gs).mean(0), poses, roads, sim.state[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--a", default="brake", help="plan A: log / brake / go / coast / a number (m/s^2)")
    p.add_argument("--b", default="go", help="plan B, same choices")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--k", type=int, default=8, help="prior samples averaged per plan (stochastic models)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sim = ImaginedDrive.load(a.ckpt, a.cache, a.data_dir, a.split)
    rc, dt = sim.cfg.render, sim.cfg.data.dt
    steps = int(round(a.seconds / dt))
    k = 1 if sim.model.cfg.deterministic else a.k
    sim.reset(a.scene, seed=a.seed); sim.save("start")
    ego = sim.ego_blue                                                    # the ego's box, same in every frame
    gA, pA, rA, vA = expected(sim, plan_value(a.a), steps, k, a.seed)
    gB, pB, rB, vB = expected(sim, plan_value(a.b), steps, k, a.seed)

    # B's prediction at step t, re-projected into A's ego frame at step t (one warp per step)
    warped = warp_previous(gB[:, None], pB, pA[:, None], rc)[:, 0, 0]
    valid = warp_previous(torch.ones_like(gB)[:, None], pB, pA[:, None], rc)[:, 0, 0] > 0.999
    # drop a 1-pixel rim: B's own crop edge lands there after the warp (interpolation against the zero padding)
    valid = -torch.nn.functional.max_pool2d(-valid.float()[:, None], 3, 1, 1)[:, 0] > 0.5
    diff = (warped - gA) * valid
    # where plan B's ego is, seen from plan A's ego (drawn as an outline in rows 3 and 4)
    egoB_in_A = warp_previous(ego.expand(steps, 1, 64, 64).clone(), pB, pA[:, None], rc)[:, 0, 0] > 0.5

    print(f"scene {a.scene} | plan A = {a.a} (ends at {vA:.1f} m/s) | plan B = {a.b} (ends at {vB:.1f} m/s) | "
          f"{'deterministic' if k == 1 else f'mean of {k} prior samples'}")
    print(f"{'time':>6s} {'|B-A| (shared area)':>20s} {'B adds':>9s} {'B removes':>10s} {'shared area':>12s}")
    for t in range(1, steps, 2):
        m = valid[t].float().sum().clamp_min(1)
        print(f"{(t + 1) * dt:5.1f}s {diff[t].abs().sum() / m:20.5f} {diff[t].clamp(min=0).sum():9.2f} "
              f"{(-diff[t]).clamp(min=0).sum():10.2f} {valid[t].float().mean():11.0%}")

    # picture: columns every 0.4 s; rows A, B, B in A's frame, difference. Rows 1-3 in the training colours (road dim
    # red, others green, ego blue). Row 4: road dim grey, ego A blue, difference red/green. White outline = plan B's
    # ego, seen from plan A's ego (rows 3 and 4).
    S, cols = 4, list(range(1, steps, 2))
    def outline(mask):
        m = mask.float()[None, None]
        return (torch.nn.functional.max_pool2d(m, 3, 1, 1) - m)[0, 0] > 0
    def up(img):
        return np.kron((img.clip(0, 1) * 255).astype(np.uint8), np.ones((S, S, 1), np.uint8))
    def tile(g, road, egoB=None):
        f = torch.stack([road * 0.35, g.clamp(0, 1), ego]).permute(1, 2, 0).numpy().copy()
        if egoB is not None:
            f[outline(egoB).numpy()] = 1.0
        return up(f)
    def diff_tile(d, road, egoB):
        img = np.repeat((road * 0.25).numpy()[..., None], 3, axis=2)
        img[..., 0] += (-d).clamp(min=0).numpy() * 2; img[..., 1] += d.clamp(min=0).numpy() * 2
        img[..., 2] += ego.numpy()
        img[outline(egoB).numpy()] = 1.0
        return up(img)
    rows = [("A: " + a.a, [tile(gA[t], rA[t]) for t in cols]),
            ("B: " + a.b, [tile(gB[t], rB[t]) for t in cols]),
            ("B in A's frame", [tile(warped[t], rA[t], egoB_in_A[t]) for t in cols]),
            ("B - A", [diff_tile(diff[t], rA[t], egoB_in_A[t]) for t in cols])]
    W, H, LW = 64 * S, 64 * S, 120
    im = Image.new("RGB", (LW + len(cols) * (W + 2), 16 + len(rows) * (H + 2)), (20, 20, 20))
    d = ImageDraw.Draw(im)
    for j, t in enumerate(cols):
        d.text((LW + j * (W + 2) + 4, 2), f"+{(t + 1) * dt:.1f}s", fill=(220, 220, 220))
    for i, (label, tiles) in enumerate(rows):
        d.text((6, 16 + i * (H + 2) + H // 2), label, fill=(230, 230, 230))
        for j, tl in enumerate(tiles):
            im.paste(Image.fromarray(tl), (LW + j * (W + 2), 16 + i * (H + 2)))
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"reaction_scene{a.scene}_{a.a}_vs_{a.b}.png")
    im.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
