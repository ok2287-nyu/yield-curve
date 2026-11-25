"""Look at closed-loop episodes: a controller vs the human on the same windows, drawn in world coordinates.

    # the v4 policy's crashes, from a replay_eval CSV
    python scripts/episode_sheet.py --cache ../round_cache --data_dir ../round_data \
        --csv runs/planner_eval/pol_replay.csv --controller pol:pol_v4 --policy_ckpts runs/pol_v4/best.pt \
        --select crash --out runs/planner_eval/sheets/pol_v4_crash.png
    # windows where it drives differently from the human without colliding (half slower, half faster)
    ... --select differ --n 8 --out runs/planner_eval/sheets/pol_v4_differ.png
    # planners: --controller mpc:ab_inv4_01 --ckpts runs/ab_inv4_01/last.pt --planners mpc ; constants: --controller coast

Each row is one window, rerun in ReplayEnv (the same seeds as replay_eval, so the same episode):
    - panels at --times (s): the road photo, the logged road users (grey boxes; dots for road users without a size),
      the human's car where the human really was (blue outline) and the controller's car (orange; red once it has
      collided). ~60 m around the two cars;
    - speed over time (human vs controller) and the gap to the car ahead (front_gap, as in the metrics).
Selection from the CSV (rows of --controller): crash = front or side collision, earliest first; differ = no collision
of any class, the largest |lag| (half where it is slower than the human, half faster). The images contain the rounD
background photo: keep them on HPC or your own machine.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from roundabout_wm.config import Config
from roundabout_wm.controller.replay import ReplayEnv, front_gap
from roundabout_wm.render import car_corners


def select(df, controller, how, n):
    d = df[df.policy == controller].copy()
    if d.empty:
        raise SystemExit(f"no rows for {controller} in the CSV (policies: {sorted(df.policy.unique())})")
    if how == "crash":
        d = d[d.col_front | d.col_side]
        d["t_hit"] = d[["t_front", "t_side"]].min(axis=1)
        return d.sort_values("t_hit").head(n)
    clean = d[~(d.col_front | d.col_side | d.col_rear) & np.isfinite(d.lag_s)]
    slow = clean.sort_values("lag_s", ascending=False).head((n + 1) // 2)
    fast = clean.sort_values("lag_s").head(n // 2)
    return pd.concat([slow, fast[~fast.window.isin(slow.window)]])


def build_controller(a, cfg):
    import replay_eval as rv
    if a.controller in rv.POLICIES:
        return rv.POLICIES[a.controller]()
    ns = argparse.Namespace(ckpts=a.ckpts, planners=a.planners, samples=a.samples, masks=a.masks, plan_T=a.plan_T,
                            w_prog=a.w_prog, w_epi=None, seed=a.seed, policy_ckpts=a.policy_ckpts)
    pols = {**rv.load_planners(ns, cfg), **rv.load_policies(ns, cfg)}
    if a.controller not in pols:
        raise SystemExit(f"{a.controller} not built; available: {sorted(pols)}")
    return pols[a.controller]


def episode(env, pol, window):
    """Run one episode, keeping the road users of every step. -> log DataFrame, list of others dicts."""
    obs = env.reset(window)
    pol.reset(env)
    others = [env.others_at(env.t)]
    while not env.done:
        obs = env.step(pol.act(env, obs))
        others.append(env.others_at(env.t))
    if hasattr(pol, "episode_summary"):
        pol.episode_summary()
    return pd.DataFrame(env.log), others


def draw_cars(ax, o, **kw):
    if not len(o["x"]):
        return
    sized = (o["length"] > 0.1) & (o["width"] > 0.1)
    if sized.any():
        cs = car_corners(o["x"][sized], o["y"][sized], o["heading"][sized], o["length"][sized], o["width"][sized])
        for c in cs:
            ax.fill(c[:, 0], c[:, 1], **kw)
    if (~sized).any():
        ax.plot(o["x"][~sized], o["y"][~sized], "o", ms=2.5, color=kw.get("fc", "0.5"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--csv", required=True, help="replay_eval output with the controller's rows")
    p.add_argument("--controller", required=True, help="policy name in the CSV, e.g. pol:pol_v4, mpc:ab_inv4_01")
    p.add_argument("--select", choices=("crash", "differ"), default="crash")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--times", default="0,1.5,3,4.5,6", help="seconds into the episode")
    p.add_argument("--split", default="val")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--reactive", action="store_true")
    p.add_argument("--ckpts", nargs="*", default=[])
    p.add_argument("--planners", default="mpc")
    p.add_argument("--policy_ckpts", nargs="*", default=[])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--masks", type=int, default=4)
    p.add_argument("--plan_T", type=int, default=10)
    p.add_argument("--w_prog", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = Config()
    dt = cfg.data.dt
    rows = select(pd.read_csv(a.csv), a.controller, a.select, a.n)
    if rows.empty:
        print(f"no {a.select} windows for {a.controller}")
        return
    cols = [c for c in ("window", "rec", "col_front", "col_side", "col_rear", "t_front", "t_side", "min_gap",
                        "progress", "lag_s", "pol_acc_mean") if c in rows]
    print(rows[cols].round(2).to_string(index=False))
    split = json.load(open(os.path.join(a.cache, "split.json")))[a.split]
    env = ReplayEnv(a.cache, a.data_dir, split, cfg, horizon=a.horizon, reactive=a.reactive)
    pol = build_controller(a, cfg)
    from uncertainty_map import background
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = [float(x) for x in a.times.split(",")]
    K = len(times)
    fig, axes = plt.subplots(len(rows), K + 1, figsize=(2.6 * K + 3.6, 2.7 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * K + [1.5]})
    bgs = {}
    for i, (_, r) in enumerate(rows.iterrows()):
        L, others = episode(env, pol, int(r.window))
        path = env.path
        if env.rec_id not in bgs:
            bgs[env.rec_id] = background(a.data_dir, env.rec_id, cfg)
        bg, bg_m = bgs[env.rec_id]
        ext = [0, bg.shape[1] * bg_m, -bg.shape[0] * bg_m, 0]
        T = int(L.t.iloc[-1])
        hit = L[L.hits.apply(len) > 0]
        t_hit = int(hit.t.iloc[0]) if len(hit) else None
        for j, ts in enumerate(times):
            ax = axes[i, j]
            t = min(int(round(ts / dt)), T)
            th = min(t, len(path.s) - 1)
            hx, hy, hh = path.xy[th, 0], path.xy[th, 1], path.heading[th]
            cx, cy, ch = L.x.iloc[t], L.y.iloc[t], L.heading.iloc[t]
            ax.imshow(bg, cmap="gray", extent=ext, alpha=0.55)
            draw_cars(ax, others[t], fc="0.35", ec="none", alpha=0.85)
            hc = car_corners(hx, hy, hh, *env.ego_size)[0]
            ax.fill(hc[:, 0], hc[:, 1], fc="none", ec="tab:blue", lw=1.6)
            crashed = t_hit is not None and t >= t_hit
            cc = car_corners(cx, cy, ch, *env.ego_size)[0]
            ax.fill(cc[:, 0], cc[:, 1], fc="tab:red" if crashed else "tab:orange", ec="k", lw=0.5, alpha=0.9)
            mx, my = (hx + cx) / 2, (hy + cy) / 2
            ax.set_xlim(mx - 30, mx + 30); ax.set_ylim(my - 30, my + 30)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{t * dt:.1f} s" + ("  (ended)" if t < round(ts / dt) else ""), fontsize=8)
        lab = (f"win {int(r.window)} rec {int(r.rec)}\n" + (f"hit at {t_hit * dt:.1f} s" if t_hit is not None
                                                             else f"lag {r.lag_s:+.1f} s, prog {r.progress:.2f}"))
        axes[i, 0].set_ylabel(lab, fontsize=7)
        ax = axes[i, K]
        tt = L.t.to_numpy() * dt
        hv = path.speed[:min(T + 1, len(path.speed))]
        ax.plot(np.arange(len(hv)) * dt, hv, color="tab:blue", lw=1.5, label="human speed")
        ax.plot(tt, L.v, color="tab:orange", lw=1.5, label=f"{a.controller} speed")
        hgap = [front_gap(env.ego_dict((path.xy[k, 0], path.xy[k, 1], path.heading[k])), others[k])
                for k in range(min(T + 1, len(path.s)))]
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(hgap)) * dt, np.minimum(hgap, 40), color="tab:blue", lw=1, ls=":", label="human gap")
        ax2.plot(tt, np.minimum(L.gap, 40), color="tab:orange", lw=1, ls=":", label="controller gap")
        ax2.set_ylim(0, 42); ax2.tick_params(labelsize=6)
        ax.set_ylim(0, max(12, float(np.nanmax(L.v)) + 1)); ax.tick_params(labelsize=6)
        if t_hit is not None:
            ax.axvline(t_hit * dt, color="tab:red", lw=1)
        if i == 0:
            ax.set_title("speed m/s (solid), gap to car ahead m (dotted, cut at 40)", fontsize=7)
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=5.5, loc="lower left", frameon=False)
    fig.suptitle(f"{a.controller}: {a.select} windows ({'reactive' if a.reactive else 'log replay'}). Blue outline = "
                 f"the human, orange = the controller (red after a collision), grey = logged road users",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
