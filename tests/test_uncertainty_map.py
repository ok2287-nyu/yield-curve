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
