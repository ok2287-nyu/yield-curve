"""scripts/uncertainty_map.py: cell -> world geometry and the per-cell maps."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roundabout_wm.config import Config
from roundabout_wm.models.world_model import WMConfig, WorldModel
from roundabout_wm.render import world_to_ego_px
from uncertainty_map import cell_world_xy, posterior_track, sample_maps


def test_cell_world_xy_inverts_the_renderer():
    rc = Config().render
    rng = np.random.default_rng(0)
    for _ in range(5):
        ego = np.array([[rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-np.pi, np.pi)]])
        xy = cell_world_xy(ego, rc)[0]                                      # (8, 8, 2)
        px = world_to_ego_px(xy, *ego[0], rc)                               # (8, 8, 2) col, row
        r = np.arange(8) * 8 + 3.5
        assert np.allclose(px[..., 0], r[None, :]) and np.allclose(px[..., 1], r[:, None])
    # heading 0 (facing +x): the top row of cells is ahead (+x), the left column is to the left (+y)
    xy = cell_world_xy(np.zeros((1, 3)), rc)[0]
    assert xy[0, 3, 0] > 25 and xy[4, 0, 1] > 20


def test_maps_shapes_and_fixed_prior():
    cfg = Config()
    n, H = cfg.data.n_cond, 3
    g = torch.Generator().manual_seed(0)
    b = {"images": (torch.rand(2, n + 5, 3, 64, 64, generator=g) > 0.9).float(),
         "actions": torch.randn(2, n + 4, 2, generator=g), "speed": torch.randn(2, n + 5, 1, generator=g)}
    for kw in ({}, {"fixed_prior": True}, {"z_global": 8}):
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=n, **kw)).eval()
        sig, kl = posterior_track(m, b, n, H)
        spread, err, occ = sample_maps(m, b, n, H, k=3)
        assert spread.shape == err.shape == occ.shape == (2, H, 8, 8) and (spread >= 0).all()
        if kw.get("z_global"):
            assert sig is None and kl is None
        else:
            assert sig.shape == kl.shape == (2, H, 8, 8) and (kl >= -1e-5).all()
            if kw.get("fixed_prior"):
                assert torch.allclose(sig, torch.ones_like(sig))


def test_script_occ_only_and_hotspots():
    """End to end on the synthetic recording: --occ_only writes the _occ files and prints the hotspot list; the
    road-user-only world map has no more samples than the all-cells one."""
    import subprocess
    import test_action_gain as tag
    S = tag._setup()
    out = os.path.join(S["root"], "unc")
    base = [sys.executable, os.path.join(ROOT, "scripts", "uncertainty_map.py"), "--cache", S["cache"],
            "--data_dir", S["raw"], "--ckpts", S["ck"], "--n_windows", "6", "--k", "2", "--batch", "3",
            "--min_count", "1", "--out_dir", out]
    env = dict(os.environ, PYTHONPATH=ROOT)
    r_all = subprocess.run(base, capture_output=True, text=True, env=env, cwd=S["root"])
    r_occ = subprocess.run(base + ["--occ_only"], capture_output=True, text=True, env=env, cwd=S["root"])
    assert r_all.returncode == 0 and r_occ.returncode == 0, r_occ.stdout[-1500:] + r_occ.stderr[-1500:]
    assert "hotspots (road-user cells" in r_occ.stdout and "sigma_p" in r_occ.stdout
    a, o = np.load(os.path.join(out, "uncertainty_fake.npz")), np.load(os.path.join(out, "uncertainty_fake_occ.npz"))
    assert 0 < o["count"].sum() < a["count"].sum()


def test_true_motion_zero_for_static_world_and_edge_keep():
    """A world that only the ego moves through has ~0 motion; a road user that moves in the world does not."""
    from uncertainty_map import edge_keep, true_motion
    from roundabout_wm.wm_eval import warp_previous
    cfg = Config()
    n, H, rc = cfg.data.n_cond, 3, cfg.render
    T = n + H
    pose = torch.zeros(1, T, 4)
    pose[0, :, 0] = torch.arange(T) * 2.0                                  # ego drives +x at 10 m/s
    g0 = torch.zeros(1, 1, 64, 64)
    g0[..., 20:24, 28:36] = 1.0                                            # a parked car ahead
    imgs = torch.zeros(1, T, 3, 64, 64)
    imgs[:, :, 1] = warp_previous(g0, pose[:, 0, :3], pose[..., :3], rc)[:, :, 0]
    b = {"images": imgs, "ego_state": pose}
    static = true_motion(b, n, H, rc)
    moved = imgs.clone()
    moved[0, n:, 1] = torch.roll(moved[0, n:, 1], 8, dims=-1)              # the car jumps sideways in the world
    mov = true_motion({"images": moved, "ego_state": pose}, n, H, rc)
    assert static.shape == (1, H, 8, 8) and static.max() < 0.05 and mov.max() > 4 * static.max()
    keep = edge_keep(2, 3, 1).reshape(2, 3, 8, 8)
    assert keep[:, :, 1:7, 1:7].all() and not keep[:, :, 0].any() and not keep[:, :, :, 7].any()
    assert edge_keep(1, 1, 0).all()


def test_script_edge_motion_fork():
    import subprocess
    import test_action_gain as tag
    S = tag._setup()
    out = os.path.join(S["root"], "unc2")
    base = [sys.executable, os.path.join(ROOT, "scripts", "uncertainty_map.py"), "--cache", S["cache"],
            "--data_dir", S["raw"], "--ckpts", S["ck"], "--n_windows", "6", "--k", "2", "--batch", "3",
            "--min_count", "1", "--out_dir", out, "--occ_only"]
    env = dict(os.environ, PYTHONPATH=ROOT)
    r0 = subprocess.run(base, capture_output=True, text=True, env=env, cwd=S["root"])
    assert r0.returncode == 0, r0.stderr[-1500:]
    # find a road-user location from the hotspot list to aim the fork at
    line = [l for l in r0.stdout.splitlines() if l.strip().startswith("sigma_p ") and "top:" in l][0]
    x, y = (float(v) for v in line.split("top: (")[1].split(")")[0].split(","))
    r = subprocess.run(base + ["--edge_cells", "1", "--motion_bins", "2", "--fork_xy", str(x), str(y),
                               "--fork_r", "6", "--fork_n", "2", "--fork_k", "3"],
                       capture_output=True, text=True, env=env, cwd=S["root"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "border dropped" in r.stdout and "motion control" in r.stdout and "with motion taken out" in r.stdout
    assert "fork / blur" in r.stdout and "centroid scatter" in r.stdout
    e = np.load(os.path.join(out, "uncertainty_fake_occ_e1.npz"))
    o = np.load(os.path.join(out, "uncertainty_fake_occ.npz"))
    assert e["count"].sum() < o["count"].sum()
    for f in ("uncertainty_fake_occ_e1_motion.png", "fork_fake_occ_e1.png"):
        assert os.path.exists(os.path.join(out, f)), f


def test_dropout_rollouts_differ_only_by_mask():
    """Same prior noise for every mask: with dropout 0 all rollouts are identical; with dropout 0.1 they differ.
    The model is left in eval mode."""
    from uncertainty_map import dropout_rollouts
    cfg = Config()
    n, H = cfg.data.n_cond, 3
    g = torch.Generator().manual_seed(0)
    b = {"images": (torch.rand(2, n + 5, 3, 64, 64, generator=g) > 0.9).float(),
         "actions": torch.randn(2, n + 4, 2, generator=g), "speed": torch.randn(2, n + 5, 1, generator=g)}
    for p, differ in ((0.0, False), (0.1, True)):
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=n, dropout=p)).eval()
        out = dropout_rollouts(m, b, n, H, 3)
        assert out.shape == (3, 2, H, 64, 64) and not m.training
        d = (out[1:] - out[0]).abs().max().item()
        assert (d > 1e-4) == differ, (p, d)


def test_script_masks_map():
    import subprocess
    import test_action_gain as tag
    S = tag._setup()
    out = os.path.join(S["root"], "unc3")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "uncertainty_map.py"), "--cache", S["cache"],
                        "--data_dir", S["raw"], "--ckpts", S["ck"], "--n_windows", "6", "--k", "2", "--batch", "3",
                        "--min_count", "1", "--out_dir", out, "--occ_only", "--masks", "2", "--motion_bins", "2"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=S["root"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "drop vs err" in r.stdout and "drop vs sigma_p" in r.stdout
    assert "drop" in np.load(os.path.join(out, "uncertainty_fake_occ.npz")).files
