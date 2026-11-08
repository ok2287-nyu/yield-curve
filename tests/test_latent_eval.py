"""scripts/latent_eval.py: per-number prior/posterior comparison and the two rollout tracks."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch

from latent_eval import car_cells, pair_stats, track
from roundabout_wm.config import Config
from roundabout_wm.models.world_model import WMConfig, WorldModel, gauss_kl


def test_pair_stats_known_values():
    z = torch.zeros(3)
    s = pair_stats(z, z, z, z)                                     # identical bells
    assert s["kl"].abs().max() < 1e-7 and s["gap"].abs().max() == 0 and torch.allclose(s["bc"], torch.ones(3))
    # q = N(1, 0.5^2), p = N(0, 1): gap 1, ratio 0.5, KL = ln 2 + (0.25 + 1)/2 - 1/2
    s = pair_stats(torch.ones(1), torch.tensor([math.log(0.25)]), torch.zeros(1), torch.zeros(1))
    assert abs(s["gap"].item() - 1) < 1e-6 and abs(math.exp(s["log_ratio"].item()) - 0.5) < 1e-6
    assert abs(s["kl"].item() - (math.log(2) + 1.25 / 2 - 0.5)) < 1e-6
    bc = math.sqrt(2 * 0.5 / 1.25) * math.exp(-1 / (4 * 1.25))
    assert abs(s["bc"].item() - bc) < 1e-6
    # summed KL matches the model's gauss_kl
    g = torch.Generator().manual_seed(0)
    a = [torch.randn(2, 8, 8, 8, generator=g) for _ in range(4)]
    assert torch.allclose(pair_stats(*a)["kl"].flatten(1).sum(1), gauss_kl(*a), atol=1e-4)


def test_car_cells():
    gr = torch.zeros(1, 1, 64, 64); gr[0, 0, 10, 50] = 1.0       # a road user in cell (row 1, col 6)
    m = car_cells(gr, (1, 8, 8, 8))
    assert m.shape == (1, 8, 8, 8) and m[:, :, 1, 6].all() and m.sum() == 8


def test_tracks_run_and_fixed_prior():
    cfg = Config()
    n, T = cfg.data.n_cond, 4
    g = torch.Generator().manual_seed(0)
    b = {"images": (torch.rand(2, n + T, 3, 64, 64, generator=g) > 0.9).float(),
         "actions": torch.randn(2, n + T - 1, 2, generator=g), "speed": torch.randn(2, n + T, 1, generator=g)}
    for kw in ({}, {"z_global": 8}, {"fixed_prior": True}, {"reencode": True, "residual": True}):
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=n, **kw)).eval()
        for drive in ("post", "prior"):
            out = track(m, b, n, T, drive)
            assert len(out) == T
            st, cars = out[0]
            assert (st["kl"] >= -1e-5).all() and ((st["bc"] > 0) & (st["bc"] <= 1 + 1e-6)).all()
            assert (cars is None) == bool(kw.get("z_global"))
        # step 1 of both tracks starts from the same h, so the comparison there is identical
        torch.manual_seed(1); a = track(m, b, n, T, "post")[0][0]["kl"]
        torch.manual_seed(1); c = track(m, b, n, T, "prior")[0][0]["kl"]
        assert torch.allclose(a, c)
