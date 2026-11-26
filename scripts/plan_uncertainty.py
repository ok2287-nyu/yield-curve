"""Is braking uncertain too? Every plan's epistemic term at the same decision, by the real gap to the car ahead.

    # 1. decisions with per-plan terms (needs PR #80: epi_<plan>, risk_<plan> in the planner trace)
    python scripts/crash_uncertainty.py --cache ../round_cache --data_dir ../round_data \
        --ckpts runs/ab_inv4_01/last.pt runs/ab_anchor/last.pt --n_windows 1000 --out runs/planner_eval/crash_unc_pp.csv
    # 2. this script (no rollouts; reads the CSVs)
    python scripts/plan_uncertainty.py --csv runs/planner_eval/crash_unc_pp.csv

At each decision the planner imagined all six plans (brake -3, gentle -1.5, coast 0, nudge +0.75, go +1.5, hold) from
the same state. For each plan: epi = std over dropout masks of its summed risk; risk = mean over masks and samples.

Decisions are grouped as
    clean      every decision in episodes without any collision;
    pre-crash  decisions within --before steps (default 5 = 1 s) before a front or side collision,
and binned by the REAL gap to the car ahead (40 = none within 40 m).

Per policy, group and gap bin it prints:
    mean epi per plan;
    brake/coast  median over decisions of epi_brake / epi_coast (1 = braking as uncertain as cruising);
    brake low    share of decisions where brake has the lowest epi of all plans;
    spread       median over decisions of (max - min) / mean of epi across plans: how much the plans differ at all
                 (near 0 = a penalty on epi shifts every plan by the same amount and cannot change the choice);
    risk=epi     share of decisions where the lowest-risk plan is also the least uncertain one;
and the same epi per plan divided by the plan's risk (relative uncertainty), since a std tends to grow with the mean.
Writes <csv stem>_by_plan.csv with the table, and a plot unless --no_plot.
"""
import argparse
import os

import numpy as np
import pandas as pd

PLANS = ("brake", "gentle", "coast", "nudge", "go", "hold")
GAP_BINS = (0, 2, 4, 6, 10, 20, 40, 41)
GAP_LABELS = ("0-2", "2-4", "4-6", "6-10", "10-20", "20-40", "none")


def label(steps, eps, before, dt=0.2):
    """steps + 'group' (clean / pre / other) and 'gap_bin'."""
    hit = eps[eps.col_front | eps.col_side].copy()
    hit["t_hit"] = hit[["t_front", "t_side"]].min(axis=1)
    clean = eps[~(eps.col_front | eps.col_side | eps.col_rear)]
    d = steps.merge(hit[["policy", "window", "t_hit"]], on=["policy", "window"], how="left")
    d["to_hit"] = np.round(d.t_hit / dt) - d.t
    d["group"] = "other"
    key = list(zip(clean.policy, clean.window))
    d.loc[pd.Series(list(zip(d.policy, d.window))).isin(key).values, "group"] = "clean"
    d.loc[(d.to_hit >= 1) & (d.to_hit <= before), "group"] = "pre"
    d["gap_bin"] = pd.cut(d.gap.clip(upper=40.0), GAP_BINS, right=False, labels=GAP_LABELS)
    return d


def summarise(g, plans):
    """One row of the table for a set of decisions g."""
    epi = g[[f"epi_{p}" for p in plans]].to_numpy(float)
    risk = g[[f"risk_{p}" for p in plans]].to_numpy(float)
    eps = 1e-9
    row = {"n": len(g)}
    for i, p in enumerate(plans):
        row[f"epi_{p}"] = epi[:, i].mean()
        row[f"rel_{p}"] = np.median(epi[:, i] / (risk[:, i] + eps))
    ib, ic = plans.index("brake"), plans.index("coast")
    row["brake_coast"] = np.median(epi[:, ib] / (epi[:, ic] + eps))
    row["brake_low"] = np.mean(epi.argmin(1) == ib)
    row["spread"] = np.median((epi.max(1) - epi.min(1)) / (epi.mean(1) + eps))
    row["risk_eq_epi"] = np.mean(epi.argmin(1) == risk.argmin(1))
    return row


def table(d, plans):
    rows = []
    for (pol, grp, gb), g in d[d.group != "other"].groupby(["policy", "group", "gap_bin"], observed=True):
        if len(g):
            rows.append({"policy": pol, "group": grp, "gap": gb, **summarise(g, plans)})
    return pd.DataFrame(rows)


def show(t, plans, min_n):
    for pol, tp in t.groupby("policy", sort=False):
        for grp in ("clean", "pre"):
            tg = tp[(tp.group == grp) & (tp.n >= min_n)]
            if tg.empty:
                continue
            title = "clean episodes" if grp == "clean" else "1 s before a front/side collision"
            print(f"\n== {pol}: {title}")
            print(f"   {'gap m':>6s} {'n':>6s} " + " ".join(f"{p:>7s}" for p in plans)
                  + f" {'brake/coast':>11s} {'brake low':>9s} {'spread':>7s} {'risk=epi':>8s}")
            for _, r in tg.iterrows():
                print(f"   {r.gap:>6s} {int(r.n):6d} " + " ".join(f"{r[f'epi_{p}']:7.3f}" for p in plans)
                      + f" {r.brake_coast:11.2f} {r.brake_low:9.2f} {r.spread:7.2f} {r.risk_eq_epi:8.2f}")
            print("   relative (median epi / risk):")
            for _, r in tg.iterrows():
                print(f"   {r.gap:>6s} {'':6s} " + " ".join(f"{r[f'rel_{p}']:7.3f}" for p in plans))


def plot(t, plans, path, min_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pols = list(dict.fromkeys(t.policy))
    fig, axes = plt.subplots(len(pols), 2, figsize=(10, 3.6 * len(pols)), squeeze=False, sharey="row")
    for i, pol in enumerate(pols):
        for j, grp in enumerate(("clean", "pre")):
            ax = axes[i, j]
            tg = t[(t.policy == pol) & (t.group == grp) & (t.n >= min_n)]
            x = np.arange(len(tg))
            for p in plans:
                ax.plot(x, tg[f"epi_{p}"], marker="o", label=p)
            ax.set_xticks(x, tg.gap.astype(str))
            ax.set_xlabel("real gap to the car ahead (m)")
            ax.set_title(f"{pol}: {'clean' if grp == 'clean' else '1 s before a collision'}", fontsize=9)
            if j == 0:
                ax.set_ylabel("epistemic (std of risk over masks)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="decisions CSV from crash_uncertainty.py")
    p.add_argument("--episodes", default=None, help="default: <csv stem>_episodes.csv")
    p.add_argument("--before", type=int, default=5)
    p.add_argument("--min_n", type=int, default=20, help="hide bins with fewer decisions")
    p.add_argument("--no_plot", action="store_true")
    a = p.parse_args()
    steps = pd.read_csv(a.csv)
    eps = pd.read_csv(a.episodes or a.csv.replace(".csv", "_episodes.csv"))
    plans = [q for q in PLANS if f"epi_{q}" in steps.columns]
    if "brake" not in plans or "coast" not in plans:
        raise SystemExit("no per-plan columns (epi_brake, epi_coast, ...): rerun crash_uncertainty.py with PR #80 merged")
    t = table(label(steps, eps, a.before), plans)
    show(t, plans, a.min_n)
    stem = a.csv[:-4]
    t.to_csv(stem + "_by_plan.csv", index=False)
    print(f"\nwrote {stem}_by_plan.csv")
    if not a.no_plot:
        plot(t, plans, stem + "_by_plan.png", a.min_n)


if __name__ == "__main__":
    main()
