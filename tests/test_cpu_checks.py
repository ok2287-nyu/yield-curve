"""lead_slope.py, crash_check.py, bootstrap_ci.py on the synthetic recording."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import test_action_gain as tag
from bootstrap_ci import window_recordings
from crash_check import measure
from lead_slope import fit_beta, pairs_from_recording
from plan_rank import NAMES, score_window
from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel
from roundabout_wm.round_io import load_recording

CFG = Config()


def test_fit_beta_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    n = 4000
    d = pd.DataFrame(dict(rec=rng.integers(0, 8, n), gap=rng.uniform(2, 30, n), v_f=rng.uniform(0, 12, n),
                          dv=rng.normal(0, 1, n), a_f=rng.normal(0, 1, n), a_l=rng.normal(0, 1, n)))
    d["dF_2"] = rng.normal(0, 2, n)
    d["dL_2"] = 0.4 * d.dF_2 + 0.05 * d.gap - 0.3 * d.dv + 0.5 * d.a_l + rng.normal(0, 0.3, n)
    b, r2, m = fit_beta(d, 2.0)
    assert abs(b - 0.4) < 0.03 and m == n and r2 > 0.5
    assert abs(fit_beta(d, 2.0, ("a_l",))[0] - 0.4) < 0.03


def test_pairs_find_the_lead_on_the_synthetic_road():
    S = tag._setup()
    rec = load_recording(S["raw"], 0, step=CFG.data.step)
    df = pd.DataFrame(pairs_from_recording(rec, CFG))
    assert set(zip(df.follower, df.lead)) == {(1, 2), (3, 1)}        # 2 has nobody ahead; 4 is parked off-road
    assert np.allclose(df[df.follower == 3].gap, 12 - 4.5, atol=0.3)
    assert df[["dF_1", "dL_1", "dF_2", "dL_2"]].abs().max().max() < 0.2  # everyone at constant speed


def test_crash_measure_gives_gain_one_and_kept_mass_on_true_frames():
    S = tag._setup()
    ck = torch.load(S["ck"])
    m = WorldModel(WMConfig(**ck["wm_cfg"])).eval()
    m.load_state_dict(ck["model"])
    env = ReplayEnv(S["cache"], S["raw"], [0], CFG, ck["stats"], horizon=10)
    from fake_round import ego_window
    w = ego_window(env)
    r = score_window(m, ck, env, w, 10, 1, 1, 0, CostConfig(), CFG, "cpu")
    ctx = dict(r[1])
    ctx["green"] = torch.from_numpy(ctx["real"]["green"].astype(np.float32) / 255)[None, None]  # the truth
    rows = pd.DataFrame(measure(env, ctx, 2, [4, 9], CFG.render))   # track 2 = the leader
    br = rows[(rows.plan == "brake") & (rows.t == 2.0)].iloc[0]
    assert abs(br.gain - 1) < 0.1 and abs(br.mass_w_ratio - 1) < 0.1 and br.true_shift_px > 5
    assert set(rows.plan) == set(NAMES)


def test_crash_check_and_bootstrap_scripts_run():
    S = tag._setup()
    run_dir = os.path.dirname(S["ck"])
    e = pd.DataFrame([dict(run="fake", window=0, plan="go", kind="seen", cls="front", track=2, t_hit=1.0)])
    e.to_csv(os.path.join(run_dir, "exploit_cases.csv"), index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "crash_check.py"), "--cache", S["cache"],
                        "--data_dir", S["raw"], "--ckpt", S["ck"], "--exploit",
                        os.path.join(run_dir, "exploit_cases.csv"), "--samples", "1", "--masks", "1"],
                       capture_output=True, text=True, cwd=S["root"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    out = pd.read_csv(os.path.join(run_dir, "crash_check.csv"))
    assert out.at_hit.any() and (out.kind == "crash").all()
    # bootstrap: windows -> recordings, and the script on a synthetic gain csv
    rec_of = window_recordings(S["cache"], "val", CFG)
    assert len(rec_of) > 0 and (rec_of == 0).all()
    g = pd.DataFrame(dict(run="fake", window=np.arange(min(20, len(rec_of))) % len(rec_of), track=2, group="lead",
                          plan="brake", t=2.0, gain=np.linspace(0.2, 0.8, min(20, len(rec_of)))))
    g.to_csv(os.path.join(run_dir, "g.csv"), index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bootstrap_ci.py"), "--cache", S["cache"],
                        "--gain", os.path.join(run_dir, "g.csv"), "--boot", "50"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "lead" in r.stdout and "brake" in r.stdout
