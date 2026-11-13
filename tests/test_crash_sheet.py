"""crash_sheet.py: case selection (--unexplained) and a smoke run on the synthetic recording."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pandas as pd

import test_action_gain as tag
from crash_sheet import pick


def test_unexplained_keeps_only_cases_without_a_lead_gain(tmp_path):
    e = pd.DataFrame(dict(run="r", window=[1, 2, 2], plan=["go", "go", "coast"], kind="seen", cls="front",
                          track=[5, 6, 6], t_hit=[1.0, 1.2, 1.4]))
    g = pd.DataFrame(dict(run="r", window=[1, 2], plan=["go", "go"], group="lead", t=[2.0, 1.0], track=[5, 6],
                          gain=[-0.3, 0.1]))
    e.to_csv(tmp_path / "e.csv", index=False); g.to_csv(tmp_path / "g.csv", index=False)
    got = pick(tmp_path / "e.csv", tmp_path / "g.csv", "r", unexplained=True)
    assert [(c[0], c[1]) for c in got] == [(2, "go"), (2, "coast")]      # window 2's gain is only at 1 s
    assert len(pick(tmp_path / "e.csv", None, "r", unexplained=False)) == 3


def test_crash_sheet_smoke():
    S = tag._setup()
    out = os.path.join(S["root"], "crash")
    env_w = 0
    tag._run("crash_sheet.py", "--ckpt", S["ck"], "--cases", f"{env_w}:go:2", f"{env_w}:brake:3",
             "--samples", "1", "--masks", "1", "--out_dir", out)
    assert sorted(f for f in os.listdir(out) if f.endswith(".png")) == ["crash_fake_w0_brake.png",
                                                                        "crash_fake_w0_go.png"]
