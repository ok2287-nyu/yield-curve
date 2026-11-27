"""controller/headway.py: the projected margin, the lead-speed estimate, and the guard in a synthetic scene."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import fake_round
from roundabout_wm.controller.headway import HeadwayGuard, LeadTracker, headway_margin
from roundabout_wm.controller.replay import Constant, ReplayEnv, run_episode

_E = {}


def _env():
    if not _E:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        env = ReplayEnv(cache, raw, [0], cfg, horizon=30)
        _E.update(env=env, w=fake_round.ego_window(env))
    return _E["env"], _E["w"]


def test_margin_by_hand():
    # no lead: always safe
    assert np.isinf(headway_margin(np.inf, 0, 8, [1.5, -3])).all()
    # stopped car 20 m ahead, ego 8 m/s, H = 4 s, d0 1, tau 0.5
    m = headway_margin(20.0, 0.0, 8.0, [0.0, -3.0])
    # coast: the gap would run out (8 m/s * 4 s = 32 m > 20 m) -> negative
    assert m[0] < 0
    # brake -3 for 2 s then 0: v 8 -> 2, 10 m travelled, then 2 m/s for 2 s -> 4 m: gap 6 m, need 1 + 0.5 * 2 = 2
    assert abs(m[1] - (20 - 10 - 4 - 2)) < 1e-9, m
    # a lead at the same speed: the gap never shrinks; margin = gap - (d0 + tau v)
    assert abs(headway_margin(10.0, 6.0, 6.0, [0.0])[0] - (10 - 1 - 3)) < 1e-9
    # horizon matters: a short look-ahead misses the stopped car's danger for coasting
    assert headway_margin(20.0, 0.0, 8.0, [0.0], H=5)[0] > 0


def test_lead_tracker_estimates_the_leaders_speed():
    """The fake leader drives at 6 m/s 15.5 m ahead (front to back). Coasting: the estimate is 6; going +1.5:
    still 6 (the gap shrinks because the ego speeds up, not because the leader slows)."""
    env, w = _env()
    for a in (0.0, 1.5):
        tr = LeadTracker()
        obs = env.reset(w)
        tr.reset(env)
        gaps, vs = [], []
        for _ in range(5):
            obs = env.step(a)
            g, v = tr.update(env)
            gaps.append(g); vs.append(v)
        assert abs(gaps[0] - 15.5) < 0.3, gaps
        assert all(abs(v - 6.0) < 0.3 for v in vs), (a, vs)


def test_guard_stops_go_from_hitting_the_leader():
    """Constant +1.5 drives into the leader; the guard lets it accelerate while that is safe, then holds it back."""
    env, w = _env()
    raw = run_episode(env, Constant(1.5, "go"), w)
    assert raw["col_front"]
    guard = HeadwayGuard(Constant(1.5, "go"))
    guard.record_trace = True
    m = run_episode(env, guard, w)
    s = guard.episode_summary()
    assert not m["col_front"] and m["min_gap"] > 1.0, m
    assert 0 < s["hw_share"] < 1, s
    tr = guard.trace
    assert tr[0]["applied"] == 1.5                               # 15.5 m at equal speeds: going is still safe at first
    assert any(r["applied"] < 1.5 for r in tr)
    assert all(r["applied"] == r["proposed"] or r["margin"] < 0 for r in tr)     # it only steps in when unsafe


def test_guard_passes_through_safe_and_human_actions():
    env, w = _env()
    g = HeadwayGuard(Constant(0.0, "coast"))
    run_episode(env, g, w)
    assert g.episode_summary()["hw_share"] == 0.0               # coasting behind an equal-speed leader is safe

    class Human:
        name = "h"

        def reset(self, env):
            pass

        def act(self, env, obs):
            return None

    m = run_episode(env, HeadwayGuard(Human()), w)
    assert not m["col_front"] and abs(m["progress"] - 1) < 1e-6
