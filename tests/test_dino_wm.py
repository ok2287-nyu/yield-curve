"""DINO-WM-style model (roundabout_wm/models/dino_wm.py) with the frozen "stub" backbone (no download): shapes, the
frozen encoder, the information flow, the state buffer, imagine() compatibility, checkpoint round trip, and short
end-to-end runs of train_wm.py and action_gain.py on the synthetic recording."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from roundabout_wm.config import Config
from roundabout_wm.controller.imagine import imagine
from roundabout_wm.models.dino_wm import DinoWM, hidden_size
from roundabout_wm.models.world_model import WMConfig, WorldModel

N, P, B = 10, 6, 2
L = N + P


def _cfg(**kw):
    return WMConfig(n_cond=N, arch="dino", dino="stub", hidden=hidden_size("stub", 4), **kw)


def _model(**kw):
    torch.manual_seed(0)
    return WorldModel(_cfg(**kw))


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"images": (torch.rand(B, L, 3, 64, 64, generator=g) > 0.9).float(),
            "actions": torch.randn(B, L - 1, 2, generator=g), "speed": torch.randn(B, L, 1, generator=g)}


def test_dispatch_and_frozen_encoder():
    m = _model()
    assert isinstance(m, DinoWM)
    b = _batch()
    loss, logs = m.loss(b, P)
    loss.backward()
    assert all(p.grad is None for p in m.encoder.parameters())
    assert not m.encoder.net.training                    # the backbone stays in eval mode under model.train()
    assert logs["feat"] > 0 and np.isfinite(logs["loss"])


def test_first_prediction_copies_the_last_frame():
    m = _model(deterministic=True).eval()
    b = _batch()
    out = m(b["images"], b["actions"], b["speed"], P, z_source="zero")
    e = m.encode(b["images"][:, :N])
    assert torch.allclose(out["feat"][:, 0], e[:, -1], atol=1e-5)     # the head starts at zero (residual)


def test_action_reaches_the_prediction_but_not_the_prior():
    m = _model().eval()
    with torch.no_grad():
        m.head[1].weight.normal_(0, 0.05)                               # make the predictor matter
    b = _batch()
    e = m.encode(b["images"][:, :N])
    h, _ = m.warmup(e, b["actions"], b["speed"])
    a1, a2 = b["actions"][:, N - 1], b["actions"][:, N - 1] + 2.0
    s = b["speed"][:, N - 1]
    z = m.zero_z(B, "cpu")
    assert not torch.allclose(m.rollout_step(h, z, a1, s)[1], m.rollout_step(h, z, a2, s)[1])
    mu1, _ = m.prior_params(m.with_action(h, a1, s))
    mu2, _ = m.prior_params(m.with_action(h, a2, s))
    assert torch.equal(mu1, mu2)


def test_rollout_step_matches_forward():
    """The planner's path (warmup + rollout_step) gives the same frames as forward() (deterministic)."""
    m = _model(deterministic=True).eval()
    with torch.no_grad():
        m.head[1].weight.normal_(0, 0.05)
    b = _batch()
    out = m(b["images"], b["actions"], b["speed"], P, z_source="zero")
    h, _ = m.warmup(m.encode(b["images"][:, :N]), b["actions"], b["speed"])
    for k in range(P):
        h, g = m.rollout_step(h, m.zero_z(B, "cpu"), b["actions"][:, N - 1 + k], b["speed"][:, N - 1 + k])
        assert torch.allclose(g, out["green"][:, k], atol=1e-5), k


def test_teacher_forcing_uses_true_history():
    m = _model(deterministic=True)
    with torch.no_grad():
        m.head[1].weight.normal_(0, 0.05)
    m.train()
    b = _batch()
    torch.manual_seed(1)
    tf = m(b["images"], b["actions"], b["speed"], P)["feat"]
    m.cfg.teacher_forcing = False
    torch.manual_seed(1)
    ol = m(b["images"], b["actions"], b["speed"], P)["feat"]
    assert torch.allclose(tf[:, 0], ol[:, 0], atol=1e-5) and not torch.allclose(tf[:, 2], ol[:, 2])


def test_imagine_with_samples_and_masks():
    cfg = Config()
    m = _model().eval()
    b = _batch()
    h, _ = m.warmup(m.encode(b["images"][:1, :N]), b["actions"][:1], b["speed"][:1])
    T = 5
    st = np.zeros((3, T + 1, 4)); st[..., 3] = 5.0
    paths = {"acc": np.zeros((3, T)), "curv": np.zeros((3, T)), "state": st}
    stats = {"action_mean": [0.0, 0.0], "action_std": [1.0, 1.0], "speed_mean": 0.0, "speed_std": 1.0}
    out = imagine(m, h, paths, stats, cfg, n_samples=2, n_masks=2)
    assert out["green"].shape == (2, 2, 3, T, 64, 64)


def test_checkpoint_round_trip():
    m = _model()
    d = vars(m.cfg)
    m2 = WorldModel(WMConfig(**d))
    m2.load_state_dict(m.state_dict())
    assert isinstance(m2, DinoWM)


def test_old_model_unchanged():
    assert type(WorldModel(WMConfig(n_cond=N))).__name__ == "WorldModel"


def test_train_and_action_gain_end_to_end():
    import fake_round
    root = tempfile.mkdtemp()
    cfg, raw, cache = fake_round.make(root)
    env = dict(os.environ, PYTHONPATH=ROOT)
    for run, extra in (("dino_det", ["--deterministic"]), ("dino_z", ["--beta", "1e-4", "--anchor_prior", "1e-6"])):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_wm.py"), "--cache", cache,
                            "--arch", "dino", "--dino", "stub", "--run", run, "--runs_dir", os.path.join(root, "runs"),
                            "--steps", "2", "--batch", "2", "--val_every", "2", "--val_batches", "1", "--workers",
                            "0", "--n_pred", "10", *extra], capture_output=True, text=True, cwd=root, env=env)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "done at step 2" in r.stdout
    ck = os.path.join(root, "runs", "dino_z", "last.pt")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "action_gain.py"), "--cache", cache,
                        "--data_dir", raw, "--ckpts", ck, "--split", "val", "--n_windows", "20", "--samples", "1",
                        "--masks", "1"], capture_output=True, text=True, cwd=root, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_hf_wrapper_gives_the_8x8_grid():
    """The transformers path: a small random Dinov2Model (no download) through HFDino + FrozenDino's resize."""
    import pytest
    tr = pytest.importorskip("transformers")
    from roundabout_wm.models.dino_wm import FrozenDino, HFDino
    cfg = tr.Dinov2Config(hidden_size=384, num_hidden_layers=2, num_attention_heads=6, intermediate_size=512,
                          patch_size=14, image_size=518)
    enc = FrozenDino("stub")
    enc.net = HFDino(tr.Dinov2Model(cfg)).eval()
    y = enc(torch.rand(3, 3, 64, 64))
    assert y.shape == (3, 384, 8, 8) and torch.isfinite(y).all()
