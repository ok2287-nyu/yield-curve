"""action_gain.py groups (lead / follower / stationary) and the gain measurement on the synthetic recording; a smoke
run of gain_sheet.py."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

import fake_round
from action_gain import groups
from roundabout_wm.controller.replay import ReplayEnv
from roundabout_wm.dataset import compute_stats
from roundabout_wm.models.world_model import WMConfig, WorldModel

_S = {}


def _setup():
    if not _S:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        stats = compute_stats(cache, [0], cfg)
        torch.manual_seed(0)
        wc = WMConfig(n_cond=cfg.data.n_cond)
        ck = os.path.join(root, "runs", "fake", "last.pt")
        os.makedirs(os.path.dirname(ck))
        torch.save({"model": WorldModel(wc).state_dict(), "wm_cfg": vars(wc), "stats": stats}, ck)
        _S.update(root=root, cfg=cfg, raw=raw, cache=cache, stats=stats, ck=ck)
    return _S


def test_groups_find_lead_follower_and_parked():
    S = _setup()
    env = ReplayEnv(S["cache"], S["raw"], [0], S["cfg"], S["stats"], horizon=10)
    env.reset(fake_round.ego_window(env))
    g = groups(env, 10)
    assert g == {2: "lead", 3: "follower", 4: "stationary"}


def _run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), "--cache", _S["cache"],
                        "--data_dir", _S["raw"], *args], capture_output=True, text=True, cwd=_S["root"])
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_sanity_real_gives_gain_one_for_the_follower():
    S = _setup()
    _run("action_gain.py", "--ckpts", S["ck"], "--split", "val", "--n_windows", "40", "--samples", "1",
         "--masks", "1", "--sanity", "real")
    d = pd.read_csv(os.path.join(os.path.dirname(S["ck"]), "action_gain_sanity_real.csv"))
    f = d[d.group == "follower"]
    assert len(f) > 0 and (f.gain - 1).abs().max() < 0.1


def test_gain_sheet_smoke():
    S = _setup()
    out = os.path.join(S["root"], "sheets")
    w = _run("action_gain.py", "--ckpts", S["ck"], "--split", "val", "--n_windows", "40", "--samples", "1",
             "--masks", "1", "--min_mass", "0")
    csv = os.path.join(os.path.dirname(S["ck"]), "action_gain.csv")
    if not (pd.read_csv(csv).group == "follower").any():               # an untrained model may lose every car
        return
    _run("gain_sheet.py", "--ckpt", S["ck"], "--csv", csv, "--group", "follower", "--plan", "brake", "--top", "2",
         "--samples", "1", "--masks", "1", "--out_dir", out)
    assert any(f.endswith(".png") for f in os.listdir(out))
