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
from roundabout_wm.wm_eval import counterfactual_accel, swap_actions, warp_previous

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



def test_counterfactual_accel_brakes_to_a_stop():
    stats = {"speed_mean": 6.0, "speed_std": 3.0, "action_mean": [0.3, 0.0], "action_std": [0.9, 0.03]}
    act = torch.randn(B, L - 1, 2)
    state = torch.zeros(B, L, 4); state[:, :, 3] = 5.0
    a2, s2 = counterfactual_accel(act, state, N_COND, stats, 0.2, -3.0)
    assert torch.equal(a2[:, :N_COND - 1], act[:, :N_COND - 1])                 # warm-up untouched
    assert torch.allclose(a2[:, N_COND - 1:, 0] * 0.9 + 0.3, torch.tensor(-3.0))  # constant -3 m/s^2
    assert torch.equal(a2[..., 1], act[..., 1])                                 # curvature as logged
    v = s2[0, :, 0] * 3.0 + 6.0
    assert torch.allclose(v[N_COND], torch.tensor(5.0 - 0.6), atol=1e-5) and v.min() >= 0


def test_deterministic_never_reads_the_future_and_has_no_kl():
    m, b = _model(deterministic=True), _batch()
    imgs = b["images"].clone()
    imgs[:, N_COND:] = float("nan")                                 # any read of a future frame would poison the output
    for src in ("mix", "post", "prior"):                            # every z_source is forced to z = 0
        out = m(imgs, b["actions"], b["speed"], N_PRED, z_source=src)
        assert torch.isfinite(out["green"]).all() and out["kl"] is None
    loss, logs = m.loss({**b, "images": torch.nan_to_num(imgs)}, N_PRED)
    loss.backward()
    assert logs["kl"] == 0.0
    assert all(p.grad is None for n in ("prior", "posterior") for p in getattr(m, n).parameters())


def test_deterministic_is_deterministic_in_eval():
    m, b = _model(deterministic=True), _batch()
    m.eval()
    torch.manual_seed(1); g1 = m(b["images"], b["actions"], b["speed"], N_PRED)["green"]
    torch.manual_seed(2); g2 = m(b["images"], b["actions"], b["speed"], N_PRED)["green"]
    assert torch.equal(g1, g2)


def test_warm_start_reproduces_stage_one_exactly():
    """Stage 2 starts where stage 1 ended: with P_z's weight at 0, any z gives the deterministic prediction."""
    det, b = _model(deterministic=True), _batch()
    for p in det.parameters():                                       # pretend stage 1 trained: perturb every weight
        p.data.add_(0.01 * torch.randn_like(p))
    torch.manual_seed(7)
    sto = WorldModel(WMConfig(n_cond=N_COND))
    copied = sto.load_warm_start(det.state_dict())
    assert not any(k.startswith(("prior.", "posterior.", "P_z.")) for k in copied)
    det.eval(); sto.eval()
    g_det = det(b["images"], b["actions"], b["speed"], N_PRED)["green"]
    for src in ("prior", "post"):
        g = sto(b["images"], b["actions"], b["speed"], N_PRED, z_source=src)["green"]
        assert torch.allclose(g, g_det, atol=1e-6), src
    assert sto.P_z.weight.abs().sum() == 0


def _stage2(pz_init_scale):
    det = _model(deterministic=True)
    for p in det.parameters():
        p.data.add_(0.01 * torch.randn_like(p))
    torch.manual_seed(7)
    sto = WorldModel(WMConfig(n_cond=N_COND, beta=0.0, p_zdrop=0.0))     # posterior steps only, no KL
    sto.load_warm_start(det.state_dict(), pz_init_scale=pz_init_scale)
    return det, sto


def test_zero_pz_blocks_the_mse_gradient_into_the_posterior():
    """The trap seen in run cnn_ws: with P_z's weight at 0, the MSE sends exactly no gradient to the posterior,
    so only the KL trains it (and collapses it onto the prior). A small random P_z opens the path."""
    for scale, expect_zero in ((0.0, True), (0.01, False)):
        _, sto = _stage2(scale)
        loss, _ = sto.loss(_batch(), N_PRED)
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in sto.posterior.parameters() if p.grad is not None)
        assert (g == 0.0) == expect_zero, (scale, g)


def test_small_pz_start_stays_close_to_stage_one():
    det, sto = _stage2(0.01)
    det.eval(); sto.eval()
    b = _batch()
    g_det = det(b["images"], b["actions"], b["speed"], N_PRED)["green"]
    g = sto(b["images"], b["actions"], b["speed"], N_PRED, z_source="prior")["green"]
    assert (g - g_det).abs().max() < 0.05 and not torch.equal(g, g_det)


def test_beta_schedule():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train_wm import beta_at
    A = type("A", (), {"beta": 1e-4, "kl_warmup_steps": 1000})
    assert beta_at(0, A) == 0 and beta_at(999, A) == 0
    assert abs(beta_at(1500, A) - 5e-5) < 1e-12 and beta_at(2000, A) == 1e-4 and beta_at(9000, A) == 1e-4
    A.beta_warmup = 1e-6                                              # small KL during the warm-up, then the ramp
    assert beta_at(0, A) == 1e-6 and abs(beta_at(1500, A) - (1e-6 + (1e-4 - 1e-6) / 2)) < 1e-12
    A.kl_warmup_steps = 0
    assert beta_at(0, A) == 1e-4


def test_open_loop_sequences_never_use_the_posterior():
    """With p_openloop = 1 every training step samples the prior: the prediction cannot depend on future frames
    (the posterior still sees them for the KL), and with the stop-gradient the MSE never reaches the posterior."""
    m, b = _model(p_openloop=1.0, p_zdrop=0.0, beta=0.0), _batch()
    imgs2 = b["images"].clone()
    imgs2[:, N_COND:] = torch.rand_like(imgs2[:, N_COND:])
    m.eval()
    torch.manual_seed(3); g1 = m(b["images"], b["actions"], b["speed"], N_PRED, z_source="mix")["green"]
    torch.manual_seed(3); g2 = m(imgs2, b["actions"], b["speed"], N_PRED, z_source="mix")["green"]
    assert torch.equal(g1, g2)
    m.train()
    loss, _ = m.loss(b, N_PRED)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.posterior.parameters() if p.grad is not None) == 0.0
    m0 = _model(p_openloop=0.0, p_zdrop=0.0)                      # control: all-posterior sequences do depend on it
    m0.eval()
    torch.manual_seed(3); h1 = m0(b["images"], b["actions"], b["speed"], N_PRED, z_source="mix")["green"]
    torch.manual_seed(3); h2 = m0(imgs2, b["actions"], b["speed"], N_PRED, z_source="mix")["green"]
    assert not torch.equal(h1, h2)


def _masks_drawn_per_module(per_sequence):
    """Train-mode forward pass; count how many distinct dropout masks each StickyDropout2d used."""
    m, b = _model(), _batch()
    m.train()
    seen = {}
    mods = [d for d in m.modules() if isinstance(d, StickyDropout2d)]
    orig = StickyDropout2d.forward

    def spy(self, x):
        y = orig(self, x)
        if self.training and self.p > 0:
            key = id(self.mask) if self.sticky else object()      # non-sticky: a fresh mask every call
            seen.setdefault(id(self), set()).add(key if self.sticky else id(key))
        return y
    StickyDropout2d.forward = spy
    try:
        if per_sequence:
            with m.fixed_dropout_masks():
                m.loss(b, N_PRED)
        else:
            m.loss(b, N_PRED)
    finally:
        StickyDropout2d.forward = orig
    return [len(seen[id(d)]) for d in mods]


def test_per_sequence_masks_in_training():
    """Decided Nov 4: in training each sequence keeps one dropout mask per layer for all steps (as in the epistemic
    estimate). Per-step mode draws a new one every call."""
    assert all(n == 1 for n in _masks_drawn_per_module(per_sequence=True))
    assert all(n > 1 for n in _masks_drawn_per_module(per_sequence=False))


def test_anchor_flags_off_change_nothing_and_on_work():
    """residual / reencode default off: same output as before. On: forward and loss run, gradients reach the
    encoder through the re-read frames, and the residual decoder carries a frame over when Dec outputs 0."""
    b = _batch()
    base = _model().eval()
    with torch.no_grad():
        g0 = base(b["images"], b["actions"], b["speed"], N_PRED, z_source="prior")["green"]
    for kw in ({"residual": True}, {"reencode": True}, {"residual": True, "reencode": True}):
        m = _model(**kw)
        m.load_state_dict(base.state_dict())
        m.train()
        loss, _ = m.loss(b, N_PRED)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if kw.get("reencode"):                                       # encoder gets gradient from rollout steps too
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
    with torch.no_grad():                                            # flags off: identical to the base model
        assert torch.equal(_model().eval()(b["images"], b["actions"], b["speed"], N_PRED, z_source="prior")["green"], g0)
    m = _model(residual=True)
    with torch.no_grad():
        for prm in m.dec[-1].parameters():
            prm.zero_()                                              # Dec logits = 0 -> output = sigmoid(prev)
        prev = (torch.rand(1, 1, 64, 64) > 0.5).float()
        out = m.decode(torch.zeros(1, 256, 8, 8), prev)
    assert torch.allclose(out, torch.sigmoid(prev))


def test_reencode_needs_future_road_frames():
    b = _batch()
    m = _model(reencode=True).eval()
    try:
        m(b["images"][:, :N_COND], b["actions"], b["speed"], N_PRED, z_source="prior")
    except ValueError:
        return
    raise AssertionError("reencode with only warm-up frames should raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
