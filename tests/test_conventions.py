"""Sign-convention tests: run with  python -m pytest tests  (or python tests/test_conventions.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from roundabout_wm.config import Config
from roundabout_wm.kinematics import extract_actions, rollout_np, step_np, step_torch
from roundabout_wm.render import render_frame, world_to_ego_px

RC = Config().render


def _blob_centre(channel):
    rows, cols = np.nonzero(channel > 64)
    return rows.mean(), cols.mean()


def test_ahead_is_up_and_left_is_left():
    # ego at origin heading 30 deg; one car 10 m ahead, one 5 m to the ego's left
    th = np.deg2rad(30)
    fwd, left = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    for offset, expect in [(10 * fwd, "up"), (5 * left, "left")]:
        ego = dict(x=0.0, y=0.0, heading=th, length=4.5, width=2.0)
        others = dict(x=np.array([offset[0]]), y=np.array([offset[1]]), heading=np.array([th]),
                      length=np.array([4.5]), width=np.array([2.0]))
        img = render_frame(ego, others, None, RC)
        r, c = _blob_centre(img[1])
        re, ce = _blob_centre(img[2])
        if expect == "up":
            assert abs(c - ce) < 1 and abs((re - r) * RC.m_per_px - 10) < 1
        else:
            assert abs(r - re) < 1 and abs((ce - c) * RC.m_per_px - 5) < 1


def test_ego_points_up():
    ego = dict(x=3.0, y=-7.0, heading=1.1, length=4.5, width=2.0)
    none = {k: np.zeros(0) for k in ("x", "y", "heading", "length", "width")}
    rows, cols = np.nonzero(render_frame(ego, none, None, RC)[2] > 64)
    assert np.ptp(rows) > np.ptp(cols)          # taller than wide = nose up


def test_projection_is_rigid():
    p = np.array([[1.0, 2.0], [4.0, -3.0]])
    px = world_to_ego_px(p, 0.5, 0.5, 0.7, RC)
    assert np.isclose(np.linalg.norm(p[0] - p[1]), np.linalg.norm(px[0] - px[1]) * RC.m_per_px)


def test_actions_invert_kinematics():
    dt, rng = 0.2, np.random.default_rng(0)
    acts = np.stack([rng.uniform(-1, 1, 30), rng.uniform(-0.05, 0.05, 30)], 1)
    states = rollout_np([0, 0, 0.3, 6.0], acts, dt)
    rec = extract_actions(states[:, 2], states[:, 3], dt, v_min=0.5, accel_clip=6, curv_clip=0.3)
    assert np.allclose(rec, acts, atol=1e-9)


def test_torch_matches_numpy():
    import torch
    s, a = np.array([1.0, -2.0, 0.4, 5.0]), np.array([0.5, 0.03])
    t = step_torch(torch.tensor(s), torch.tensor(a, requires_grad=True), 0.2)
    assert np.allclose(t.detach().numpy(), step_np(s, a, 0.2))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
