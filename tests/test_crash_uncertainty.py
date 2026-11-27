"""scripts/crash_uncertainty.py and the planner's per-decision trace."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from crash_uncertainty import matched_auroc, report


def _toy():
    """Window 0 hits at 2.0 s (step 10); window 1 is clean. In window 0, epi is high in the 5 decisions before the
    hit; the real gap is small there in both windows (so closeness alone cannot separate them)."""
    rows = []
    for w in (0, 1):
        for t in range(12):
            pre = w == 0 and 5 <= t <= 9
            rows.append(dict(policy="p", window=w, t=t, pick="coast", epi=1.0 if pre else 0.1, epi_pix=0.5,
                             risk=0.2, gap=3.0 if t >= 5 else 20.0, closeness=-(3.0 if t >= 5 else 20.0)))
    eps = pd.DataFrame([dict(policy="p", window=0, col_front=True, col_side=False, col_rear=False, t_front=2.0,
                             t_side=np.nan),
                        dict(policy="p", window=1, col_front=False, col_side=False, col_rear=False, t_front=np.nan,
                             t_side=np.nan)])
    return pd.DataFrame(rows), eps


def test_report_separates_signal_from_closeness():
    steps, eps = _toy()
    out = report(steps, eps, "p", before=5)
    assert out["n_hit"] == 1 and out["n_pre"] == 5
    assert out["auroc_epi"] == 1.0 and out["auroc_matched_epi"] == 1.0
    assert abs(out["auroc_matched_closeness"] - 0.5) < 1e-9          # same gap in the matched bin: no information
    assert abs(out["auroc_matched_epi_pix"] - 0.5) < 1e-9


def test_planner_trace_and_script_smoke():
    import test_planner as tp
    S = tp._setup()
    from roundabout_wm.controller.planner import ModelPlanner
    from roundabout_wm.controller.replay import run_episode
    pol = ModelPlanner(S["m"], S["stats"], S["cfg"], samples=1, masks=2)
    pol.record_trace = True
    met = run_episode(S["env"], pol, S["w"])
    tr = pd.DataFrame(pol.trace)
    assert len(tr) == met["steps"] and (tr[["epi", "epi_pix", "risk"]] >= 0).all().all()
    out = os.path.join(S["root"], "cu.csv")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "crash_uncertainty.py"), "--cache", S["cache"],
                        "--data_dir", S["raw"], "--ckpts", S["ck"], "--n_windows", "3", "--horizon", "6",
                        "--samples", "1", "--masks", "2", "--out", out],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=S["root"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "== mpc:fake" in r.stdout and os.path.exists(out) and os.path.exists(out.replace(".csv", "_episodes.csv"))


def test_fallback_match_fires_on_the_same_share():
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from fallback_match import match
    rng = np.random.default_rng(0)
    risk = rng.exponential(1.0, 10000)
    d = pd.DataFrame({"risk": risk, "epi": 0.05 * risk + rng.normal(0, 0.02, 10000)})
    share, thr, ov = match(d, 0.1)
    assert abs((d.risk > thr).mean() - share) < 0.002 and 0 < share < 0.5
    assert 0.5 < ov <= 1.0                                    # correlated triggers mostly agree
