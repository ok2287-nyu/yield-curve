"""scripts/ladder.py: car-by-car scoring (kept / faded / split / invented), the corridor geometry and the rungs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import pandas as pd
import torch

from ladder import RUNGS, SUMS, owner_map, rungs, scheduled_rollout, score_step, summarize, window_geometry
from roundabout_wm.config import Config
from roundabout_wm.models.world_model import WMConfig, WorldModel


def _box(r0, r1, c0, c1, v=1.0):
    a = np.zeros((64, 64), np.float32); a[r0:r1, c0:c1] = v
    return a


def _score(pred, fps, cors, seen=None, lead=None):
    own = owner_map(fps, cors)
    cars = [dict(seen=True if seen is None else seen[i], lead=bool(lead and lead[i]),
                 area=np.array([f.sum()]), sq_t=np.array([(f * f).sum()])) for i, f in enumerate(fps)]
    acc = {k: np.zeros(1) for k in SUMS}
    score_step(pred, own, cars, 0, acc, thr=0.3, min_px=3, min_area=4.0)
    return summarize(acc, 0)


def test_kept_faded_split_invented():
    fp = _box(20, 26, 30, 33)                                  # one car, 18 px
    cor = _box(10, 36, 28, 35) > 0                             # its corridor along its path
    s = _score(fp.copy(), [fp], [cor])
    assert abs(s["kept"] - 1) < 1e-6 and s["split"] == 0 and s["lost"] == 0 and s["invented"] == 0
    assert abs(s["sharp"] - 1) < 1e-6
    s = _score(0.3 * fp, [fp], [cor])                          # faded to 30 %
    assert abs(s["kept"] - 0.3) < 1e-6 and s["faded"] == 1
    two = _box(12, 16, 30, 33) + _box(28, 32, 30, 33)          # drawn twice (two timings) inside the corridor
    s = _score(two, [fp], [cor])
    assert s["split"] == 1 and s["kept"] > 0.9
    s = _score(fp + _box(50, 56, 5, 8), [fp], [cor])           # plus a car where nobody is
    assert abs(s["kept"] - 1) < 1e-6 and abs(s["invented"] - 1) < 1e-6


def test_overlapping_corridors_go_to_the_nearest_car():
    fa, fb = _box(20, 26, 30, 33), _box(30, 36, 30, 33)        # two cars in a platoon, corridors overlap
    ca, cb = _box(10, 40, 28, 35) > 0, _box(15, 45, 28, 35) > 0
    s = _score(fa + fb, [fa, fb], [ca, cb])
    assert abs(s["kept"] - 1) < 1e-6 and s["split"] == 0      # each car keeps its own blob, no false split
    s = _score(fa, [fa, fb], [ca, cb], lead=[False, True])     # b (the lead) vanished: its region is empty
    assert abs(s["kept"] - 0.5) < 1e-6 and s["lost"] == 0.5 and s["lead_kept"] == 0


def test_arrivals_are_not_counted_as_invented():
    fa, fb = _box(20, 26, 30, 33), _box(2, 8, 30, 33)          # b enters the view later
    ca, cb = _box(18, 28, 28, 35) > 0, _box(0, 12, 28, 35) > 0
    s = _score(fa + fb, [fa, fb], [ca, cb], seen=[True, False])
    assert s["invented"] == 0 and abs(s["arrivals"] - 1) < 1e-6 and abs(s["kept"] - 1) < 1e-6


def _window():
    """Ego drives +y at 5 m/s. 1 = leader 10 m ahead (same speed), 2 = car 12 m to the right, leaving sideways,
    3 = enters from far ahead after the warm-up."""
    cfg = Config()
    n, T, dt = cfg.data.n_cond, cfg.data.n_pred, cfg.data.dt
    st = np.zeros((n + T, 4)); st[:, 2] = np.pi / 2
    st[:, 1] = (np.arange(n + T) - (n - 1)) * 5 * dt
    win = []
    for j in range(n + T):
        t = (j - (n - 1)) * dt
        rows = {1: (0.0, 10 + 5 * t, np.pi / 2), 2: (12 + 6 * t, 0.0, 0.0)}
        if t > 1.0:
            rows[3] = (0.0, 45 - 2 * t, -np.pi / 2)
        win.append(pd.DataFrame([dict(trackId=k, x=x, y=y, heading=h, length=4.5, width=1.8)
                                 for k, (x, y, h) in rows.items()]).set_index("trackId"))
    meta = pd.DataFrame({"class": ["car", "car", "truck"]}, index=pd.Index([1, 2, 3], name="trackId"))
    return cfg, win, st, meta


def test_window_geometry_seen_lead_arrival():
    cfg, win, st, meta = _window()
    owners, cars = window_geometry(win, st, cfg, meta, tol=5, dilate=2)
    by = {c["id"]: c for c in cars}
    assert by[1]["seen"] and by[1]["lead"] and by[2]["seen"] and not by[2]["lead"]
    assert not by[3]["seen"] and by[3]["cls"] == "truck"
    assert len(owners) == cfg.data.n_pred and by[1]["area"].min() > 4            # the leader stays in view
    # scoring the truth itself: everything kept, nothing split or invented
    acc = {k: np.zeros(cfg.data.n_pred) for k in SUMS}
    from ladder import render_poses
    n = cfg.data.n_cond
    for t in range(cfg.data.n_pred):
        j = n + t
        truth = render_poses([(r.x, r.y, r.heading, r.length, r.width) for r in win[j].itertuples()], st[j], cfg.render)
        score_step(truth, owners[t], cars, t, acc, 0.3, 3, 4.0)
    for t in range(cfg.data.n_pred):
        s = summarize(acc, t)
        assert abs(s["kept"] - 1) < 1e-3 and s["split"] == 0 and s["invented"] < 1e-3 and abs(s["lead_kept"] - 1) < 1e-3


def test_rungs_shapes_and_deterministic_post_equals_prior():
    cfg = Config()
    n, T = cfg.data.n_cond, 4
    g = torch.Generator().manual_seed(0)
    b = {"images": (torch.rand(2, n + T, 3, 64, 64, generator=g) > 0.9).float(),
         "actions": torch.randn(2, n + T - 1, 2, generator=g), "speed": torch.randn(2, n + T, 1, generator=g)}
    for kw in ({"deterministic": True}, {}, {"z_global": 8}, {"reencode": True}, {"z_seq": 8}, {"z_seq": 8, "seq_prior": True}):
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=n, **kw)).eval()
        out = rungs(m, b, n, T, k=2)
        if kw.get("z_seq"):
            assert set(out) == {"truth", "recon", "roll_post", "roll_prior", "roll_pmean"}
            continue
        assert set(out) == set(RUNGS) and len(out["roll_prior"]) == 2 and len(out["mix_L1"]) == 2
        for r, preds in out.items():
            for p in preds:
                assert p.shape == (2, T, 64, 64), (kw, r, p.shape)
        if kw.get("deterministic"):
            assert torch.allclose(out["1step_post"][0], out["1step_prior"][0])
            assert torch.allclose(out["roll_post"][0], out["roll_prior"][0])
            # the first target: one-step and rollout both start from h_{n-1} after the real warm-up
            assert torch.allclose(out["1step_post"][0][:, 0], out["roll_post"][0][:, 0], atol=1e-5)
            assert not torch.allclose(out["1step_post"][0][:, 1:], out["roll_post"][0][:, 1:])



def test_scheduled_rollout_matches_forward():
    """All-posterior / all-prior schedules reproduce forward(z_source=...) exactly (same random numbers)."""
    cfg = Config()
    n, T = cfg.data.n_cond, 5
    g = torch.Generator().manual_seed(1)
    b = {"images": (torch.rand(2, n + T, 3, 64, 64, generator=g) > 0.9).float(),
         "actions": torch.randn(2, n + T - 1, 2, generator=g), "speed": torch.randn(2, n + T, 1, generator=g)}
    for kw in ({}, {"z_global": 8}, {"reencode": True, "residual": True}, {"deterministic": True}):
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=n, **kw)).eval()
        for src in ("post", "prior"):
            torch.manual_seed(3); ref = m(b["images"], b["actions"], b["speed"], T, z_source=src)["green"][:, :, 0]
            torch.manual_seed(3); got = scheduled_rollout(m, b, n, T, lambda k: src)
            assert torch.allclose(ref, got, atol=1e-6), (kw, src)
        mean = scheduled_rollout(m, b, n, T, lambda k: "pmean")
        assert torch.allclose(mean, scheduled_rollout(m, b, n, T, lambda k: "pmean"))     # no sampling
        if not kw:                                                                         # mixed schedule differs
            torch.manual_seed(3); mix = scheduled_rollout(m, b, n, T, lambda k: "post" if k % 2 == 1 else "prior")
            assert not torch.allclose(mix, ref)


if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f(); print("ok", f.__name__)
