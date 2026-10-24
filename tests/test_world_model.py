"""World-model v1 tests: encoder contract, shapes, the decided information flow, stop-gradient, warp baseline.

    python -m pytest tests          or          python tests/test_world_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from roundabout_wm.config import Config
from roundabout_wm.models.encoders import build_encoder
from roundabout_wm.models.world_model import StickyDropout2d, WMConfig, WorldModel
from roundabout_wm.render import render_frame
from roundabout_wm.wm_eval import swap_actions, warp_previous

N_COND, N_PRED, B = 10, 6, 2
L = N_COND + N_PRED


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = (torch.rand(B, L, 3, 64, 64, generator=g) > 0.9).float()
    return {"images": imgs, "actions": torch.randn(B, L - 1, 2, generator=g),
            "speed": torch.randn(B, L, 1, generator=g)}


def _model(**kw):
    torch.manual_seed(0)
    return WorldModel(WMConfig(n_cond=N_COND, **kw))


def check_encoder_contract(enc):
    """Any encoder added to build_encoder must pass this: one frame (B, 3, 64, 64) -> (B, 256, 8, 8), and
    deterministic in train mode (no dropout in the encoder, decision 17)."""
    x = torch.rand(3, 3, 64, 64)
    enc.train()
    y1, y2 = enc(x), enc(x)
    assert y1.shape == (3, 256, 8, 8), y1.shape
    assert torch.equal(y1, y2), "encoder is stochastic in train mode: no dropout in the encoder (decision 17)"


def test_encoder_contract_cnn():
    check_encoder_contract(build_encoder("cnn"))


def test_forward_shapes_all_modes():
    m, b = _model(), _batch()
    for src in ("mix", "post", "prior"):
        out = m(b["images"], b["actions"], b["speed"], N_PRED, z_source=src)
        assert out["green"].shape == (B, N_PRED, 1, 64, 64)
        assert out["green_warm"].shape == (B, N_COND, 1, 64, 64)
        assert out["prior_std_map"].shape == (B, N_PRED, 8, 8)
        assert (out["kl"] is None) == (src == "prior")
    loss, logs = m.loss(b, N_PRED)
    loss.backward()
    assert np.isfinite(loss.item())


def _prior_grad_norm(prior_sg):
    m, b = _model(beta=0.0, p_zdrop=1.0, prior_sg=prior_sg), _batch()   # every step samples the prior; no KL grad
    loss, _ = m.loss(b, N_PRED)
    loss.backward()
    return sum(float(p.grad.abs().sum()) for p in m.prior.parameters() if p.grad is not None)


def test_stop_gradient_cuts_mse_path_into_prior():
    assert _prior_grad_norm(prior_sg=True) == 0.0
    assert _prior_grad_norm(prior_sg=False) > 0.0


def test_prior_still_learns_from_kl_with_stop_gradient():
    m, b = _model(beta=1.0, p_zdrop=1.0, prior_sg=True), _batch()
    loss, _ = m.loss(b, N_PRED)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.prior.parameters()) > 0


def test_action_enters_only_pred():
    """act[i] changes the prediction of frame i+1, but not the prior or posterior at step i (only from i+1 on)."""
    m, b = _model(), _batch()
    m.eval()
    z = torch.randn(N_PRED, B, 8, 8, 8)
    act2 = b["actions"].clone()
    act2[:, N_COND - 1] += 3.0                                      # change only the first rollout action
    o1 = m(b["images"], b["actions"], b["speed"], N_PRED, z_source="given", z_seq=z)
    o2 = m(b["images"], act2, b["speed"], N_PRED, z_source="given", z_seq=z)
    assert torch.allclose(o1["prior_std_map"][:, 0], o2["prior_std_map"][:, 0])     # prior at step i: no act[i]
    assert not torch.allclose(o1["green"][:, 0], o2["green"][:, 0])                 # frame i+1 depends on act[i]
    assert not torch.allclose(o1["prior_std_map"][:, 1], o2["prior_std_map"][:, 1])  # prior reacts one step later
    p1 = m(b["images"], b["actions"], b["speed"], N_PRED, z_source="post")
    p2 = m(b["images"], act2, b["speed"], N_PRED, z_source="post")
    assert torch.allclose(p1["kl"][:, 0], p2["kl"][:, 0])                           # posterior at step i: no act[i]


def test_prior_rollout_never_sees_future_frames():
    m, b = _model(), _batch()
    m.eval()
    imgs2 = b["images"].clone()
    imgs2[:, N_COND:] = torch.rand_like(imgs2[:, N_COND:])
    torch.manual_seed(5); o1 = m(b["images"], b["actions"], b["speed"], N_PRED, z_source="prior")
    torch.manual_seed(5); o2 = m(imgs2, b["actions"], b["speed"], N_PRED, z_source="prior")
    assert torch.equal(o1["green"], o2["green"])


def test_fixed_dropout_masks_are_reused():
    """Inside one fixed_dropout_masks block the model is one deterministic ensemble member (z held fixed)."""
    m, b = _model(), _batch()
    m.train()
    z = torch.randn(N_PRED, B, 8, 8, 8)
    run = lambda: m(b["images"], b["actions"], b["speed"], N_PRED, z_source="given", z_seq=z)["green"]
    with m.fixed_dropout_masks():
        g1, g2 = run(), run()
    assert torch.equal(g1, g2)
    with m.fixed_dropout_masks():
        g3 = run()
    assert not torch.equal(g1, g3)                                  # a new block = a new mask
    assert all(d.mask is None and not d.sticky for d in m.modules() if isinstance(d, StickyDropout2d))


def test_warp_matches_renderer_for_static_cars():
    """Cars that stand still in the world: warping frame A by the ego's motion must reproduce frame B."""
    rc = Config().render
    rng = np.random.default_rng(0)
    others = {"x": rng.uniform(-15, 15, 8), "y": rng.uniform(-15, 15, 8), "heading": rng.uniform(0, 6.28, 8),
              "length": np.full(8, 4.5), "width": np.full(8, 1.9)}
    pa = np.array([1.0, -2.0, 0.6])
    pb = np.array([pa[0] + 1.8 * np.cos(0.6), pa[1] + 1.8 * np.sin(0.6), 0.6 + np.deg2rad(4)])   # 1.8 m, 4 deg
    ego = lambda p: dict(x=p[0], y=p[1], heading=p[2], length=4.5, width=1.9)
    fa = torch.from_numpy(render_frame(ego(pa), others, None, rc)[1:2]).float()[None] / 255
    fb = torch.from_numpy(render_frame(ego(pb), others, None, rc)[1:2]).float() / 255
    w = warp_previous(fa, torch.tensor(pa)[None], torch.tensor(pb)[None, None], rc)[0, 0]
    err_warp, err_copy = ((w - fb) ** 2).mean().item(), ((fa[0] - fb) ** 2).mean().item()
    assert err_warp < 0.1 * err_copy, (err_warp, err_copy)
    same = warp_previous(fa, torch.tensor(pa)[None], torch.tensor(pa)[None, None], rc)[0, 0]
    assert torch.allclose(same, fa[0], atol=1e-5)                                 # identity pose = identity


def test_swap_actions_keeps_warmup_and_kinematics():
    stats = {"speed_mean": 6.0, "speed_std": 3.0, "action_mean": [0.3, 0.0], "action_std": [0.9, 0.03]}
    g = torch.Generator().manual_seed(1)
    act = torch.randn(B, L - 1, 2, generator=g)
    state = torch.zeros(B, L, 4); state[:, :, 3] = 5.0
    spd = ((state[:, :, 3:4] - 6.0) / 3.0)
    a2, s2 = swap_actions(act, spd, state, N_COND, stats, 0.2)
    assert torch.equal(a2[:, :N_COND - 1], act[:, :N_COND - 1])
    assert torch.equal(a2[0, N_COND - 1:], act[1, N_COND - 1:])
    assert torch.equal(s2[:, :N_COND], spd[:, :N_COND])
    v = s2[0, :, 0] * 3.0 + 6.0
    acc = a2[0, :, 0] * 0.9 + 0.3
    assert torch.allclose(v[N_COND], torch.clamp(v[N_COND - 1] + acc[N_COND - 1] * 0.2, min=0), atol=1e-5)



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
