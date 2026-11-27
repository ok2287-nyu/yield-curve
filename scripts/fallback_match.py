"""Risk threshold for the control fallback, matched to the uncertainty fallback's firing rate.

    python scripts/fallback_match.py --csv runs/planner_eval/crash_unc_pp.csv --epi 0.1

Reads the decisions CSV of crash_uncertainty.py (planners WITHOUT fallback; columns policy, epi, risk of the picked
= cheapest plan). Per planner: the share of decisions with epi > --epi (how often the uncertainty fallback would
fire) and the risk threshold that fires on the same share (the (1 - share) quantile of the picked plan's risk).
Prints one line per planner, "<run> <share> <risk threshold>", e.g. "ab_inv4_01 0.082 3.41", plus the overlap of
the two triggers (share of decisions where both fire, of those where either does).
"""
import argparse

import numpy as np
import pandas as pd


def match(d, epi):
    """d: decisions of one planner -> (share, risk threshold, overlap)."""
    fire_epi = d.epi.to_numpy() > epi
    share = float(fire_epi.mean())
    thr = float(np.quantile(d.risk.to_numpy(), 1 - share)) if share > 0 else float("inf")
    fire_risk = d.risk.to_numpy() > thr
    either = (fire_epi | fire_risk).sum()
    overlap = float((fire_epi & fire_risk).sum() / either) if either else float("nan")
    return share, thr, overlap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--epi", type=float, default=0.1)
    a = p.parse_args()
    d = pd.read_csv(a.csv)
    for pol, g in d.groupby("policy", sort=False):
        share, thr, ov = match(g, a.epi)
        print(f"{pol.split(':', 1)[-1].split('@')[0]} {share:.4f} {thr:.6g} overlap={ov:.2f}")


if __name__ == "__main__":
    main()
