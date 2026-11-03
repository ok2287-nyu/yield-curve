"""Crop geometry of scripts/fov_popin.py: a car at a known offset is visible in exactly the expected crops."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

from fov_popin import crops, visible_mask
from roundabout_wm.config import Config
from roundabout_wm.render import car_corners, render_frame


def _visible(fwd, left, heading=0.7):
    """Which geometries see a 4.5 x 1.9 m car at (fwd, left) metres from an ego at the origin with this heading."""
    c, s = np.cos(heading), np.sin(heading)
    x, y = fwd * c - left * s, fwd * s + left * c
    cor = car_corners(x, y, heading, 4.5, 1.9)
    return {g: bool(visible_mask(cor, 0.0, 0.0, heading, *geo)[0]) for g, geo in crops(Config()).items()}


def test_geometries():
    assert _visible(20, 0) == {"current": True, "ego_lower": True, "crop_96m": True}
    assert _visible(38, 0) == {"current": False, "ego_lower": True, "crop_96m": True}
    assert _visible(60, 0) == {"current": False, "ego_lower": False, "crop_96m": True}
    assert _visible(-12, 0) == {"current": True, "ego_lower": False, "crop_96m": True}
    assert _visible(0, 30) == {"current": False, "ego_lower": False, "crop_96m": True}


def test_current_crop_matches_renderer():
    """The 'current' rectangle must agree with what render.py actually draws (green pixels > 0)."""
    rc = Config().render
    ahead, behind, half = crops(Config())["current"]
    ego = dict(x=0.0, y=0.0, heading=0.0, length=4.5, width=1.9)
    for fwd, left in [(ahead + 2.0, 0), (ahead + 2.4, 0), (-behind - 2.0, 0), (0, half + 2.0), (10, 0)]:
        cor = car_corners(fwd, left, 0.0, 4.5, 1.9)
        drawn = render_frame(ego, dict(x=np.array([fwd]), y=np.array([left]), heading=np.zeros(1),
                                       length=np.array([4.5]), width=np.array([1.9])), None, rc)[1].max() > 0
        assert bool(visible_mask(cor, 0, 0, 0, ahead, behind, half)[0]) == drawn, (fwd, left)


if __name__ == "__main__":
    test_geometries(); print("ok test_geometries")
    test_current_crop_matches_renderer(); print("ok test_current_crop_matches_renderer")
