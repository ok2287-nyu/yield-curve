"""Kinematic baselines: CV is exact for straight constant-speed motion, CTRV for constant-speed circles."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from roundabout_wm.baselines import extrapolate, yaw_rate


def test_cv_is_exact_for_a_straight_line():
    x, y, h = extrapolate([1.0], [2.0], [0.5], [8.0], [0.0], 3.0, turn=False)
    assert np.allclose([x[0], y[0], h[0]], [1 + 24 * np.cos(0.5), 2 + 24 * np.sin(0.5), 0.5])


def test_ctrv_is_exact_on_a_circle_and_cv_is_not():
    R, v = 30.0, 7.0                                   # a car circling a 30 m ring at 7 m/s, CCW
    w = v / R
    pos = lambda t: (R * np.cos(w * t), R * np.sin(w * t), w * t + np.pi / 2)
    x0, y0, h0 = pos(0.0)
    for t in (1.0, 2.0, 4.0):
        xt, yt, _ = pos(t)
        xa, ya, ha = extrapolate([x0], [y0], [h0], [v], [w], t, turn=True)
        xc, yc, _ = extrapolate([x0], [y0], [h0], [v], [w], t, turn=False)
        assert np.hypot(xa[0] - xt, ya[0] - yt) < 1e-9
        assert np.hypot(xc[0] - xt, yc[0] - yt) > 0.1 * v * t * w * t   # straight line leaves the circle
    assert np.isclose(yaw_rate(h0 + 0.4 * w, h0, 0.4), w)


def test_yaw_rate_wraps():
    assert np.isclose(yaw_rate(np.pi - 0.05, -np.pi + 0.05, 0.4), -0.1 / 0.4)



def test_blur_keeps_mass_and_identity():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("eb", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                                                     "eval_baselines.py"))
    eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
    x = np.zeros((2, 64, 64), np.float32); x[:, 30:34, 28:36] = 1.0
    assert np.array_equal(eb.blur(x, 0), x)
    for sg in (0.5, 2.0, 4.0):
        y = eb.blur(x, sg)
        assert y.shape == x.shape and np.isclose(y.sum(), x.sum(), rtol=1e-4)   # interior: mass kept
        assert y.max() < x.max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
