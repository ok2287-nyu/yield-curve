"""leakage.classify: which road users the ego cannot affect within the horizon."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import pandas as pd
import torch

from leakage import classify
from roundabout_wm.config import Config


def _scene():
    """Ego drives along +y from the origin at 5 m/s (plan A) or 8 m/s (plan B). Road users, per frame:
    1 'leaving'  starts 12 m to the right and 10 m ahead, drives away to the right   -> receding
    2 'follower' 10 m behind the ego in its lane, same speed                         -> not (paths overlap)
    3 'crosser'  20 m ahead, 15 m to the left, crossing the ego's route              -> not
    4 'leader'   15 m ahead in the ego's lane, faster                                -> not (ego reaches its path)"""
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    st = np.zeros((n + T, 4)); st[:, 2] = np.pi / 2
    st[:n, 1] = (np.arange(n) - (n - 1)) * 5 * dt
    win = []
    for j in range(n + T):
        t = (j - (n - 1)) * dt
        rows = {1: (12 + 10 * t, 10.0, 0.0), 2: (0.0, -10 + 5 * t, np.pi / 2),
                3: (-15 + 8 * t, 20.0, 0.0), 4: (0.0, 15 + 9 * t, np.pi / 2)}
        win.append(pd.DataFrame([dict(trackId=k, x=x, y=y, heading=h, length=4.5, width=1.8)
                                 for k, (x, y, h) in rows.items()]).set_index("trackId"))
    pose = lambda v: torch.tensor([[0.0, v * (t + 1) * dt, np.pi / 2] for t in range(T)])
    return cfg, win, st, pose(5.0), pose(8.0)


def test_only_the_leaving_car_is_receding():
    cfg, win, st, pA, pB = _scene()
    seen, rec = classify(win, st, pA, pB, cfg, tol=0.2, clear=8.0)
    assert {1, 2, 3, 4} <= seen, seen
    assert rec == [1], rec


if __name__ == "__main__":
    test_only_the_leaving_car_is_receding(); print("ok test_only_the_leaving_car_is_receding")
