"""scripts/yield_starts.py: finding stopped road users, deciding when they start, AUROC."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from roundabout_wm.config import Config
from roundabout_wm.wm_eval import warp_previous
from yield_starts import auroc, stopped_users


def _scene(move_at):
    """Ego drives +x at 5 m/s; two parked cars ahead. Car B (right) is removed from frame n + move_at on."""
    cfg = Config()
    n, F, rc = cfg.data.n_cond, 12, cfg.render
    T = n + F
    pose = torch.zeros(T, 3)
    pose[:, 0] = torch.arange(T) * 1.0
    a = torch.zeros(64, 64); a[22:26, 20:28] = 1.0
    bm = torch.zeros(64, 64); bm[22:26, 40:48] = 1.0
    warp = lambda x: warp_previous(x[None, None], pose[n - 1][None], pose[None], rc)[0, :, 0]
    ga, gb = warp(a), warp(bm)
    gb[n + move_at:] = 0.0
    return (ga + gb).clamp(0, 1), pose, n, F, rc


def test_stopped_users_finds_both_and_times_the_start():
    green, pose, n, F, rc = _scene(move_at=3)
    us = sorted(stopped_users(green, pose, n, F, rc), key=lambda u: u["mask"].nonzero()[:, 1].float().mean())
    assert len(us) == 2
    assert us[0]["start"] == -1 and us[1]["start"] == 3
    assert us[0]["fp"].shape == (F, 64, 64) and us[0]["area"] == 32


def test_moving_road_user_is_not_stopped():
    """Car A drives sideways in the world at ~2.7 m/s before n: not stopped. Car B (parked) still is."""
    green, pose, n, F, rc = _scene(move_at=100)
    for t in range(n):
        a = torch.zeros(64, 64); a[22:26, 20 + t:28 + t] = 1.0                  # 1 px = 0.75 m per 0.2 s
        bm = torch.zeros(64, 64); bm[22:26, 40:48] = 1.0
        w = warp_previous((a + bm)[None, None], pose[0][None], pose[t][None, None], rc)[0, 0, 0]
        green[t] = w.clamp(0, 1)
    us = stopped_users(green, pose, n, F, rc)
    assert len(us) >= 1 and all(u["mask"].nonzero()[:, 1].float().mean() > 36 for u in us)


def test_real_renders_parked_and_starting_cars():
    """Frames from the real renderer: a car parked 4 m left of the road, and one that waits 4 m to the right and
    drives off at 6 m/s at t = 4.0 s. Ego at 6 m/s; the window whose last observed frame is t = 2.8 s. Frame n+k is
    at t = 3.0 + 0.2 k; the waiting car has moved 1.35 m (30 % of 4.5 m) after t = 4.23 s, so start k = 7."""
    import tempfile
    import fake_round
    from roundabout_wm.dataset import RounDWindows
    from roundabout_wm.dataset import compute_stats
    root = tempfile.mkdtemp()
    go = lambda t: ((0.0, 0.0) if t < 4.0 else (6.0 * (t - 4.0), 6.0))
    cfg, raw, cache = fake_round.make(root, extra=((5, 40.0, -46.0, lambda t: (0.0, 0.0)), (6, 44.0, -54.0, go)))
    ds = RounDWindows(cache, [0], cfg, compute_stats(cache, [0], cfg), n_pred=cfg.data.n_pred)
    n, rc = cfg.data.n_cond, cfg.render
    found = None
    for i in range(len(ds)):
        it = ds[i]
        x_last = float(it["ego_state"][n - 1, 0])
        if abs(x_last - (10.0 + V_EGO * 2.8)) < 0.05 and abs(float(it["ego_state"][n - 1, 1]) + 50) < 0.1:
            found = it
            break
    assert found is not None
    green, pose = found["images"][:, 1].float(), found["ego_state"][:, :3]
    us = stopped_users(green, pose, n, cfg.data.n_pred, rc)
    side = {("left" if u["mask"].nonzero()[:, 1].float().mean() < rc.ego_col else "right"): u for u in us}
    assert set(side) == {"left", "right"}, [u["start"] for u in us]
    assert side["left"]["start"] == -1
    assert side["right"]["start"] == 7


V_EGO = 6.0


def test_auroc():
    assert auroc([2, 3], [0, 1]) == 1.0 and auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5 and np.isnan(auroc([], [1]))


def test_script_smoke():
    """End to end on a fake recording with a parked and a starting car, untrained model: runs, finds both kinds,
    prints the AUROC lines, writes the CSV and the plot."""
    import tempfile
    import fake_round
    import pandas as pd
    from roundabout_wm.dataset import compute_stats
    from roundabout_wm.models.world_model import WMConfig, WorldModel
    root = tempfile.mkdtemp()
    go = lambda t: ((0.0, 0.0) if t < 4.0 else (6.0 * (t - 4.0), 6.0))
    cfg, raw, cache = fake_round.make(root, extra=((5, 40.0, -46.0, lambda t: (0.0, 0.0)), (6, 44.0, -54.0, go)))
    torch.manual_seed(0)
    wc = WMConfig(n_cond=cfg.data.n_cond)
    ck = os.path.join(root, "runs", "fake", "last.pt")
    os.makedirs(os.path.dirname(ck))
    torch.save({"model": WorldModel(wc).state_dict(), "wm_cfg": vars(wc), "stats": compute_stats(cache, [0], cfg)}, ck)
    out = os.path.join(root, "ys")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "yield_starts.py"), "--cache", cache,
                        "--data_dir", raw, "--ckpts", ck, "--n_windows", "40", "--batch", "4",
                        "--region", "42", "-50", "10", "--out_dir", out],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=root)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "AUROC start_2s vs stays" in r.stdout and "aligned to the start" in r.stdout, r.stdout[-1500:]
    d = pd.read_csv(os.path.join(out, "yield_starts_fake.csv"))
    assert {"stays", "start_2s"} <= set(d.label)
    assert os.path.exists(os.path.join(out, "yield_starts.png"))
