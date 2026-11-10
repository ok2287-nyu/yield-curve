"""scripts/closed_mass.py: the closed-window test."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import pandas as pd

from closed_mass import is_closed
from fov_popin import crops
from roundabout_wm.config import Config


def _win(extra=None):
    """Ego drives +y at 5 m/s; road user 1 follows 10 m ahead at the same speed. extra(t) -> optional 2nd user."""
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    st = np.zeros((n + T, 4)); st[:, 2] = np.pi / 2
    st[:, 1] = (np.arange(n + T) - (n - 1)) * 5 * dt
    win = []
    for j in range(n + T):
        t = (j - (n - 1)) * dt
        rows = [dict(trackId=1, x=0.0, y=10 + 5 * t, heading=np.pi / 2, length=4.5, width=1.8)]
        if extra is not None and extra(t) is not None:
            rows.append(dict(trackId=2, **extra(t), heading=0.0, length=4.5, width=1.8))
        win.append(pd.DataFrame(rows).set_index("trackId"))
    return cfg, win, st


def test_closed_and_open():
    cfg, win, st = _win()
    box = crops(cfg)["current"]
    assert is_closed(win, st, cfg.data.n_cond, box, 1.0) == (True, 1)
    # a car enters from the side after 2 s -> open
    cfg, win, st = _win(lambda t: dict(x=60 - 10 * t, y=st_y(t)) )
    assert not is_closed(win, st, cfg.data.n_cond, box, 1.0)[0]
    # a car in view at the start leaves sideways -> open
    cfg, win, st = _win(lambda t: dict(x=5 + 6 * t, y=st_y(t)))
    assert not is_closed(win, st, cfg.data.n_cond, box, 1.0)[0]
    # empty crop -> not counted (min_users = 1)
    cfg, win, st = _win()
    win = [w.iloc[0:0] for w in win]
    assert not is_closed(win, st, cfg.data.n_cond, box, 1.0)[0]


def st_y(t):
    return 5 * t
