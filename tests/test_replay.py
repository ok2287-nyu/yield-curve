"""controller/replay.py: geometry, the human acceptance test, and collision classes on a synthetic recording."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import fake_round
import pandas as pd

from roundabout_wm.controller.replay import (Constant, HumanAccel, HumanExact, HumanPath, ReactiveSim, ReplayEnv,
                                             front_gap, lane_gaps, rects_overlap, run_episode)
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
    assert next(h for h in out["hit_ids"][1] if h) == [(2, "front")]  # it first drives into the leader
    # without followers: the follower (12 m behind) is gone, the leader stays
    assert out["green"][0, 0, 50:].sum() > 0 and out["green_nf"][0, 0, 50:].sum() == 0
    assert np.array_equal(out["green"][0, 0, :40], out["green_nf"][0, 0, :40])
    fp = env.footprint(2, out["poses"][0, 1], 1)                    # the leader alone, 20 m ahead = 26.7 px up
    assert abs(fp.sum() - (4.5 / 0.75) * (1.8 / 0.75)) < 2.0 and fp[43 - 27].sum() > 0



# ---- reactive road users ------------------------------------------------------------------------------------------
_RENV = {}


def _renv():
    if not _RENV:
        env, w = _env()
        _RENV["env"] = ReplayEnv(env.cache_dir, env.data_dir, [0], env.cfg, horizon=30, reactive=True)
    return _RENV["env"], _env()[1]


def test_reactive_human_replays_exactly():
    env, w = _renv()
    obs = env.reset(w)
    n = env.cfg.data.n_cond
    cached = (env.item["images"].numpy() * 255).round().astype(np.uint8)
    t = 0
    while not env.done and n + t + 1 < len(cached):
        obs = env.step(None); t += 1
        assert np.abs(obs["image"].astype(int) - cached[n - 1 + t]).max() <= 1, t
        assert env.sim.delayed() == []
    m = run_episode(env, HumanExact(), w)
    assert not (m["col_front"] or m["col_side"] or m["col_rear"]), m


def test_reactive_follower_brakes_with_the_ego():
    env, w = _renv()
    brake = run_episode(env, Constant(-3.0), w)          # pure replay: the follower drives into it (test above)
    assert not brake["col_rear"] and not brake["col_front"], brake
    assert 3 in env.sim.delayed() and 2 not in env.sim.delayed()      # the leader does not react to the ego behind
    go = run_episode(env, Constant(1.5), w)              # the leader ahead does not yield: still a front collision
    assert go["col_front"], go


def test_reactive_open_loop():
    env, w = _renv()
    env.reset(w)
    T = 20                                               # 4 s: long enough for the logged follower to reach the ego
    speeds = np.stack([np.full(T + 1, 6.0), np.maximum(6.0 - 3.0 * 0.2 * np.arange(T + 1), 0.0),
                       6.0 + 2.0 * np.arange(T + 1)])
    out = env.open_loop(speeds)
    plain, _ = _env()
    plain.reset(w)
    ref = plain.open_loop(speeds)
    assert np.array_equal(out["green"][0], ref["green"][0]) and out["delayed"][0][-1] == []   # human speed: = log
    assert ref["rear"][1].any() and not out["rear"][1].any() and 3 in out["delayed"][1][-1]   # brake: no rear-end
    assert out["front"][2].any()                                                               # go: still crashes
    # footprint of the follower under the brake plan follows the simulation, not the log
    k = T
    rows = np.arange(64)[:, None]
    a = env.footprint(3, out["poses"][1, k], k, plan=1)
    b = env.footprint(3, out["poses"][1, k], k)
    assert (a * rows).sum() / a.sum() > (b * rows).sum() / b.sum() + 3   # simulated: further behind (lower in frame)


class _StubRec:
    """Straight road: ego 1 stands at x = 0; road users drive along +x at 6 m/s through it in the log."""

    def __init__(self, starts, classes=None, ego_x=0.0):
        rows, meta = [], []
        for tid, x0 in starts.items():
            v = 0.0 if tid == 1 else 6.0
            x00 = ego_x if tid == 1 else x0
            for k in range(40):
                rows.append(dict(trackId=tid, frame=5 * k, x=x00 + v * 0.2 * k, y=0.0, heading=0.0, speed=v))
            meta.append(dict(trackId=tid, length=4.5, width=1.8, **{"class": (classes or {}).get(tid, "car")}))
        tr = pd.DataFrame(rows)
        self.tracks = {t: g.reset_index(drop=True) for t, g in tr.groupby("trackId")}
        self.meta = pd.DataFrame(meta).set_index("trackId")


def _run_stub(R, T=30):
    ego = dict(x=0.0, y=0.0, heading=0.0, length=4.5, width=1.8)
    sim = ReactiveSim(R, 1, 0, 5, T, np.tile([1e4, 0.0, 0.0], (T + 1, 1)), 0.2)   # logged ego far away
    out = []
    for _ in range(T):
        sim.advance(ego, 0.0)
        out.append(sim.others())
    return sim, out


def test_stub_chain_of_followers_stops_without_overlap():
    # ego log = standing ego: the log itself has the followers drive through it. Tolerance reference = that log, so
    # here we give the sim a logged ego far away instead: they must stop behind the standing ego, the second behind
    # the first (it reacts to a delayed road user, not to the ego it cannot see)
    R = _StubRec({1: 0.0, 2: -15.0, 3: -27.0})
    sim = ReactiveSim(R, 1, 0, 5, 30, np.tile([1e4, 0.0, 0.0], (31, 1)), 0.2)
    ego = dict(x=0.0, y=0.0, heading=0.0, length=4.5, width=1.8)
    for _ in range(30):
        sim.advance(ego, 0.0)
        o = sim.others()
        x = dict(zip(o["trackId"].tolist(), o["x"].tolist()))
        assert x[2] < -4.5 and x[3] < x[2] - 4.5, x               # nobody overlaps
    assert x[3] > -27.0 + 6.0 * 0.2 * 30 - 200                    # (it moved at all)
    assert sorted(sim.delayed()) == [2, 3] and abs(sim.agents[0].v) < 0.5


def test_stub_logged_tailgating_is_not_braked_and_pedestrians_do_not_react():
    # a follower 5 m behind a car that drives exactly as in the log: IDM's desired gap (~8 m) is larger, but it
    # accepted 5 m in the log, so it is not slowed
    R = _StubRec({1: 0.0, 2: -5.0, 4: -20.0}, classes={4: "pedestrian"})
    ego_log = np.stack([np.zeros(31), np.zeros(31), np.zeros(31)], 1)
    for k in range(31):
        ego_log[k, 0] = 0.0
    # the "ego" here drives at 6 m/s exactly as its log
    R.tracks[1] = R.tracks[1].assign(x=6.0 * 0.2 * np.arange(40), speed=6.0)
    ego_log[:, 0] = 6.0 * 0.2 * np.arange(31)
    sim = ReactiveSim(R, 1, 0, 5, 30, ego_log, 0.2)
    for t in range(30):
        sim.advance(dict(x=ego_log[t, 0], y=0.0, heading=0.0, length=4.5, width=1.8), 6.0)
        assert sim.delayed() == [], t
    # the ego stops: the car brakes, the pedestrian walks through (non-reactive, in the log)
    sim, out = _run_stub(_StubRec({1: 0.0, 2: -15.0, 4: -20.0}, classes={4: "pedestrian"}))
    assert 2 in sim.delayed() and 4 not in sim.delayed()


def test_lane_gaps_matches_front_gap():
    a = dict(x=0.0, y=0.0, heading=0.0, length=4.5, width=1.8)
    b = dict(x=np.array([10.0, 10.0, -10.0]), y=np.array([0.0, 5.0, 0.0]), heading=np.zeros(3),
             length=np.full(3, 4.5), width=np.full(3, 1.8))
    g = lane_gaps(a, b)
    assert abs(g[0] - front_gap(a, {k: v[:1] for k, v in b.items()})) < 1e-9 and np.isinf(g[1:]).all()



def test_plan_rank_reference_column():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from plan_rank import ref_col
    d = pd.DataFrame(dict(run=["a_react", "a_react"], real_risk=[1.0, 2.0], real_risk_nf=[0.0, 0.0]))
    assert ref_col(d) == "real_risk" and ref_col(d.assign(run="a")) == "real_risk_nf"


def test_plan_rank_reactive_cache_restores_plan_road_users():
    env, w = _renv()
    env.reset(w)
    speeds = np.stack([np.full(21, 6.0), np.maximum(6.0 - 0.6 * np.arange(21), 0.0)])
    real = env.open_loop(speeds)
    env.reset(w)                                            # a second checkpoint: reset forgets, use_plans restores
    env.use_plans(real)
    assert np.array_equal(env.footprint(3, real["poses"][1, 20], 20, plan=1),
                          env.footprint(3, real["poses"][1, 20], 20, plan=1)) and env._plan_others is not None


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
