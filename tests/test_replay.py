"""controller/replay.py: geometry, the human acceptance test, and collision classes on a synthetic recording."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import fake_round
from roundabout_wm.controller.replay import (Constant, HumanAccel, HumanExact, HumanPath, ReplayEnv, front_gap,
                                             rects_overlap, run_episode)
from roundabout_wm.render import car_corners

_ENV = {}


def _env():
    if not _ENV:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        env = ReplayEnv(cache, raw, [0], cfg, horizon=30)
        _ENV["env"], _ENV["w"] = env, fake_round.ego_window(env)
    return _ENV["env"], _ENV["w"]


def test_rects_overlap():
    a = car_corners(0, 0, 0, 4.5, 1.8)[0]
    b = car_corners([4.0, 4.6, 0.0, 3.0], [0.0, 0.0, 1.85, 1.0], [0.0, 0.0, 0.0, np.pi / 4], 4.5, 1.8)
    # touching nose to tail at 4.0 m overlaps; 4.6 m apart does not; side by side 1.85 m apart does not; rotated
    # car whose corner reaches into the ego does
    assert rects_overlap(a, b).tolist() == [True, False, False, True]


def test_front_gap_sees_a_car_across_the_lane():
    ego = dict(x=0.0, y=0.0, heading=0.0, length=4.5, width=1.8)
    across = dict(x=np.array([10.0]), y=np.array([0.0]), heading=np.array([np.pi / 2]), length=np.array([4.5]),
                  width=np.array([1.8]))
    assert abs(front_gap(ego, across) - (10 - 0.9 - 2.25)) < 1e-6
    beside = dict(across, y=np.array([5.0]), heading=np.array([0.0]))
    assert front_gap(ego, beside) == np.inf


def test_path_pose_and_stops():
    xy = np.array([[0, 0], [1, 0], [1, 0], [1, 0], [2, 0]], float)
    p = HumanPath(xy, np.zeros(5), np.array([5.0, 0.0, 0.0, 0.0, 5.0]))
    assert p.length == 2.0 and np.allclose(p.pose(1.5)[:2], (1.5, 0))
    assert p.human_speed_at(np.array([1.0]))[0] == 0.0           # it stood at s = 1
    assert p.time_at(2.0, 0.2) == 0.8


def test_human_replays_the_cache_exactly_and_never_collides():
    env, w = _env()
    obs = env.reset(w)
    n = env.cfg.data.n_cond
    cached = (env.item["images"].numpy() * 255).round().astype(np.uint8)
    assert np.abs(obs["image"].astype(int) - cached[n - 1]).max() <= 1
    pol = HumanExact()
    t = 0
    while not env.done and n + t + 1 < len(cached):
        obs = env.step(pol.act(env, obs)); t += 1
        assert np.abs(obs["image"].astype(int) - cached[n - 1 + t]).max() <= 1, t
        assert np.allclose(obs["action"], env.ds[w]["actions"][n - 2 + t].numpy(), atol=1e-4)
    m = run_episode(env, HumanExact(), w)
    assert not (m["col_front"] or m["col_side"] or m["col_rear"]), m
    assert abs(m["min_gap"] - (20 - 4.5)) < 1e-6 and m["freeze_s"] == 0 and abs(m["progress"] - 1) < 1e-6


def test_human_accel_and_constant_plans():
    env, w = _env()
    m = run_episode(env, HumanAccel(), w)
    assert not m["col_front"] and abs(m["progress"] - 1) < 1e-6
    go = run_episode(env, Constant(1.5), w)                          # closes 16.5 m on the leader: drives into it
    assert go["col_front"] and not go["col_rear"], go
    brake = run_episode(env, Constant(-3.0), w)                      # stops; the logged follower drives into it
    assert brake["col_rear"] and not brake["col_front"], brake
    assert brake["hard_brakes"] == 0 and brake["progress"] < 0.5    # -3 is not "harder than 3"
    env.stop_on_collision = False                                   # let the follower pass through: it stands
    try:
        brake = run_episode(env, Constant(-3.0), w)
    finally:
        env.stop_on_collision = True
    assert brake["freeze_s"] >= 3.0 and brake["steps"] == 30, brake


def test_open_loop_plans():
    env, w = _env()
    env.reset(w)
    T = 10
    speeds = np.stack([np.full(T + 1, 6.0), 6.0 + 2.0 * np.arange(T + 1)])
    out = env.open_loop(speeds)
    assert out["green"].shape == (2, T, 64, 64)
    assert not out["front"][0].any() and out["front"][1].any()
    assert np.allclose(out["gap"][0], 15.5)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
