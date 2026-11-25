"""scripts/episode_sheet.py: selection from a replay_eval CSV and an end-to-end sheet on the synthetic recording."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import fake_round
from episode_sheet import select


def test_select_crash_and_differ():
    d = pd.DataFrame(dict(policy=["p"] * 6 + ["q"], window=range(7),
                          col_front=[1, 0, 0, 0, 0, 1, 1], col_side=[0, 1, 0, 0, 0, 0, 0], col_rear=[0, 0, 1, 0, 0, 0, 0],
                          t_front=[3.0, np.nan, np.nan, np.nan, np.nan, 1.0, 0.2],
                          t_side=[np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                          lag_s=[0, 0, 5, 2.0, -1.5, 0, 0])).astype({"col_front": bool, "col_side": bool, "col_rear": bool})
    assert select(d, "p", "crash", 5).window.tolist() == [5, 1, 0]          # earliest hit first; rear excluded
    assert select(d, "p", "differ", 2).window.tolist() == [3, 4]            # clean only: slowest, then fastest


def test_sheet_end_to_end():
    """replay_eval (human, coast, go) -> sheets of go's crashes and coast's differing windows; also on --split test
    (the fake recording is in every split)."""
    root = tempfile.mkdtemp()
    cfg, raw, cache = fake_round.make(root)
    env = dict(os.environ, PYTHONPATH=ROOT)
    csv = os.path.join(root, "re.csv")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_eval.py"), "--cache", cache,
                        "--data_dir", raw, "--n_windows", "6", "--horizon", "30", "--policies", "human,coast,go",
                        "--split", "test", "--out", csv], capture_output=True, text=True, env=env, cwd=root)
    assert r.returncode == 0, r.stdout[-1000:] + r.stderr[-2000:]
    d = pd.read_csv(csv)
    assert (d[d.policy == "go"].col_front | d[d.policy == "go"].col_side).any()     # go hits the lead
    for ctrl, how in (("go", "crash"), ("coast", "differ")):
        out = os.path.join(root, f"{ctrl}.png")
        s = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "episode_sheet.py"), "--cache", cache,
                            "--data_dir", raw, "--csv", csv, "--controller", ctrl, "--select", how, "--n", "2",
                            "--split", "test", "--out", out], capture_output=True, text=True, env=env, cwd=root)
        assert s.returncode == 0, s.stdout[-1000:] + s.stderr[-2500:]
        assert os.path.exists(out), s.stdout
