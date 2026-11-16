"""The predict-then-warp baseline (controller/warp_plan.py) and its --warp_ref flag in the evaluation scripts."""
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
import roundabout_wm.controller.warp_plan as wp
from fake_round import ego_window
from plan_rank import NAMES
from roundabout_wm.config import Config
from roundabout_wm.controller.imagine import ego_paths, imagine, route_from_item, warm_states
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.models.world_model import WMConfig, WorldModel

CFG = Config()


def _setup():
    S = tag._setup()
    ck = torch.load(S["ck"])
    m = WorldModel(WMConfig(**ck["wm_cfg"])).eval()
    m.load_state_dict(ck["model"])
    env = ReplayEnv(S["cache"], S["raw"], [0], CFG, ck["stats"], horizon=10)
    w = ego_window(env)
    env.reset(w)
    it = env.item
    state0, rs, rk, log_acc = route_from_item(it, ck["stats"], CFG)
    accels = np.stack([log_acc[:10]] + [np.full(10, v) for v in (-3.0, -1.5, 0.0, 1.5)])
    paths = ego_paths(state0, rs, rk, accels, CFG)
    h, prev = warm_states(m, [it], CFG, "cpu")
    return S, ck, m, env, paths, h, prev


def test_reference_plan_is_reproduced_exactly():
    S, ck, m, env, paths, h, prev = _setup()
    out = wp.warp_imagine(m, h, paths, ck["stats"], CFG, 2, 1, seed=3, prev0=prev, ref="coast", names=tuple(NAMES))
    sub = {k: paths[k][[3]] for k in ("acc", "curv", "state")}
    ref = imagine(m, h, sub, ck["stats"], CFG, 2, 1, seed=3, prev0=prev)["green"]
    assert out["green"].shape == (1, 2, 5, 10, 64, 64)
    assert (out["green"][:, :, 3] - ref[:, :, 0]).abs().max() < 1e-4


def test_warped_true_frames_match_the_other_plans():
    """Feed the TRUE coast frames as the 'reference prediction': warped to brake / go they must match the true
    brake / go frames (everyone in the synthetic scene is replayed, i.e. non-reactive)."""
    S, ck, m, env, paths, h, prev = _setup()
    real = env.open_loop(paths["state"][:, :, 3])
    G = torch.from_numpy(real["green"].astype(np.float32) / 255)                      # (P, T, 64, 64)
    orig = wp.imagine
    try:
        wp.imagine = lambda *a, **k: {"green": G[3][None, None, None]}               # (1, 1, 1, T, 64, 64)
        out = wp.warp_imagine(m, h, paths, ck["stats"], CFG, 1, 1, ref="coast", names=tuple(NAMES))
    finally:
        wp.imagine = orig
    for p in (1, 4):                                                                    # brake, go
        warped, true = out["green"][0, 0, p], G[p]
        err = (warped - true).abs()[:, 4:-12, 4:-4].sum((-1, -2))    # away from the edge strips
        moved = (G[3] - true).abs()[:, 4:-12, 4:-4].sum((-1, -2))
        assert moved[-1] > 10 and (err[-1] < 0.25 * moved[-1])


def test_scripts_run_with_warp_ref():
    S = tag._setup()
    for script, extra, csv in (("plan_rank.py", ["--warp_ref", "mix"], "plan_rank_warpmix.csv"),
                               ("action_gain.py", ["--warp_ref", "coast", "--min_mass", "0"],
                                "action_gain_warpcoast.csv")):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), "--cache", S["cache"],
                            "--data_dir", S["raw"], "--ckpts", S["ck"], "--split", "val", "--n_windows", "6",
                            "--samples", "1", "--masks", "1", *extra], capture_output=True, text=True, cwd=S["root"])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        d = pd.read_csv(os.path.join(os.path.dirname(S["ck"]), csv))
        assert d.run.str.endswith("_warp" + extra[1]).all()
