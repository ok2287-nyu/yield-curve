"""roundabout_wm/controller: kinematic plans along the route, batched imagined rollouts, and the cost."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from roundabout_wm.config import Config
from roundabout_wm.controller.cost import CostConfig, ego_footprint, plan_costs, proximity_mask, step_risk
from roundabout_wm.controller.imagine import ego_paths, imagine, route_from_item
from roundabout_wm.kinematics import rollout_np
from roundabout_wm.models.world_model import WMConfig, WorldModel

CFG = Config()


def test_ego_paths_follow_route_by_distance():
    rs = np.array([0.0, 10.0, 20.0, 30.0]); rk = np.array([0.0, 0.05, 0.0])   # turn between 10 and 20 m
    st0 = np.array([0.0, 0.0, 0.0, 5.0])
    p = ego_paths(st0, rs, rk, np.array([[0.0] * 10, [-3.0] * 10]), CFG)
    assert p["state"].shape == (2, 11, 4)
    assert np.allclose(p["state"][0, :, 3], 5.0) and p["state"][1, -1, 3] == 0.0           # brake: stops, no reverse
    s = np.concatenate([[0], np.cumsum(5.0 * 0.2 * np.ones(9))])                            # coast: distance at each step
    assert np.allclose(p["curv"][0], np.where((s >= 10) & (s < 20), 0.05, 0.0))
    assert (p["curv"][1][p["state"][1, :-1, 3] < CFG.data.v_min_curvature] == 0).all()     # no turning when stopped


def _model(**kw):
    torch.manual_seed(0)
    return WorldModel(WMConfig(n_cond=CFG.data.n_cond, **kw)).eval()


def _item():
    """A consistent fake window: ego states from integrating random actions, normalised as in the dataset."""
    n, T = CFG.data.n_cond, CFG.data.n_pred
    rng = np.random.default_rng(0)
    raw = np.stack([rng.uniform(-1, 1, n + T - 1), rng.uniform(-0.03, 0.03, n + T - 1)], -1)
    st = rollout_np(np.array([0.0, 0.0, 0.3, 6.0]), raw, CFG.data.dt)
    stats = {"action_mean": [0.0, 0.0], "action_std": [1.0, 0.05], "speed_mean": 6.0, "speed_std": 2.0}
    g = torch.Generator().manual_seed(1)
    item = {"images": (torch.rand(n + T, 3, 64, 64, generator=g) > 0.9).float(),
            "actions": torch.tensor((raw - 0.0) / np.array([1.0, 0.05]), dtype=torch.float32),
            "speed": torch.tensor((st[:, 3:4] - 6.0) / 2.0, dtype=torch.float32),
            "ego_state": torch.tensor(st, dtype=torch.float32)}
    return item, stats


def test_imagine_log_plan_matches_the_model_on_logged_actions():
    """Deterministic model, the human's accelerations along the route = model.forward with the logged actions."""
    n, T = CFG.data.n_cond, 6
    item, stats = _item()
    m = _model(deterministic=True)
    state0, rs, rk, log_acc = route_from_item(item, stats, CFG)
    paths = ego_paths(state0, rs, rk, log_acc[None, :T], CFG)
    b = {k: v[None] for k, v in item.items()}
    h, _ = m.warmup(m.encode(b["images"][:, :n]), b["actions"], b["speed"])
    out = imagine(m, h, paths, stats, CFG, n_samples=4, n_masks=1)
    ref = m(b["images"], b["actions"], b["speed"], T, z_source="zero")["green"][:, :, 0]
    assert out["green"].shape == (1, 1, 1, T, 64, 64)
    assert torch.allclose(out["green"][0, 0], ref, atol=2e-4), (out["green"][0, 0] - ref).abs().max()


def test_masks_shared_across_plans_and_eps_across_masks():
    item, stats = _item()
    state0, rs, rk, _ = route_from_item(item, stats, CFG)
    paths = ego_paths(state0, rs, rk, np.zeros((2, 4)), CFG)                 # two identical plans
    h = torch.randn(1, 256, 8, 8)
    det = imagine(_model(deterministic=True), h, paths, stats, CFG, n_masks=3)["green"]          # (3, 1, 2, T)
    assert torch.equal(det[:, 0, 0], det[:, 0, 1])                           # same mask -> identical plans agree
    assert not torch.allclose(det[0], det[1])                                # different masks differ
    sto = _model()
    g0 = imagine(sto, h, paths, stats, CFG, n_samples=2, n_masks=1, seed=5)["green"]
    g1 = imagine(sto, h, paths, stats, CFG, n_samples=2, n_masks=1, seed=5)["green"]
    assert torch.equal(g0, g1) and not sto.training                          # seeded, model left in eval mode
    gm = imagine(sto, h, paths, stats, CFG, n_samples=2, n_masks=3, seed=5)["green"]
    assert gm.shape == (3, 2, 2, 4, 64, 64) and not sto.training


def test_cost_terms():
    rc, cc = CFG.render, CostConfig()
    F = ego_footprint(rc, cc)
    assert abs(F.sum().item() - (4.5 / 0.75) * (1.8 / 0.75)) < 2.0
    g = torch.zeros(1, 1, 1, 3, 64, 64)
    g[..., 0, 38:49, 28:36] = 1.0                                            # step 0: a car covering the ego
    g[..., 1, 30:33, 30:33] = 1.0                                            # step 1: 8-10 m ahead
    g[..., 2, 50:55, 30:33] = 1.0                                            # step 2: behind the ego
    col, prox = step_risk(g, torch.tensor([[0.0, 8.0, 8.0]]), rc, cc)
    assert col[..., 0] > 0.95 and col[..., 1] == 0 and col[..., 2] == 0
    assert prox[..., 0] > 0.95 and 0 < prox[..., 1] < 0.5 and prox[..., 2] == 0
    _, slow = step_risk(g, torch.tensor([[0.0, 0.0, 0.0]]), rc, cc)          # at standstill the zone is shorter
    assert slow[..., 1] < prox[..., 1]
    m = proximity_mask(torch.tensor([3.0, 12.0]), rc, cc)
    assert m.shape == (2, 64, 64) and (m[1] >= m[0]).all()
    c = plan_costs(g, np.array([[8.0, 8.0, 8.0, 8.0]]), np.zeros((1, 3)), np.zeros((1, 3)), rc, cc)
    assert c["epistemic"].item() == 0 and abs(c["progress"].item() - 3 * (1 - 0.8) ** 2) < 1e-5
    assert c["total"].item() > c["risk"].item() > 0


def test_predicted_gap_and_headway():
    from roundabout_wm.controller.cost import headway_violation, predicted_gap
    rc, cc = CFG.render, CostConfig()
    g = torch.zeros(3, 64, 64)
    g[0, 20:26, 29:34] = 1.0             # a car ahead in the lane: rows 20..25 -> nearest row 25
    g[1, 20:26, 5:10] = 1.0              # a car ahead in another lane: not in the corridor
    g[2, 40:46, 29:34] = 1.0             # a car on the ego
    gap = predicted_gap(g, rc, cc)
    assert abs(gap[0] - ((43 - 25) * 0.75 - 2.25)) < 1e-5 and gap[1] == float("inf") and gap[2] == 0
    # a lead at 10 m that the model lets vanish while the ego drives at 8 m/s
    seq = torch.tensor([[10.0, 9.0, float("inf"), float("inf")]])
    v = torch.full((1, 4), 8.0)
    bad, _ = headway_violation(seq, v, d0=1.0, tau=1.0)
    assert not bad[0]                    # 10, 9 >= 1 + 8, then the lead vanishes: not seen
    bad, capped = headway_violation(seq, v, d0=1.0, tau=1.0, lead_vmax=6.0)
    assert bad[0] and abs(capped[0, -1] - 8.2) < 1e-5     # the lead can pull away at 6 m/s at most: 9 - 2 * 0.4
