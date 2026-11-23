"""controller/policy.py and scripts/train_policy.py: the policy rollout matches the planner's imagination, gradients
reach the policy, the uncertainty cost, and an end-to-end train -> replay run."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import fake_round
from roundabout_wm.controller.cost import CostConfig
from roundabout_wm.controller.imagine import ego_paths, imagine
from roundabout_wm.controller.policy import (A_HI, A_LO, LearnedPolicy, PolicyNet, policy_rollout, rollout_cost,
                                             routes_from_batch)
from roundabout_wm.controller.replay import ReplayEnv, run_episode
from roundabout_wm.dataset import RounDWindows, compute_stats
from roundabout_wm.models.world_model import GREEN, WMConfig, WorldModel

_S = {}


def _setup():
    if not _S:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        stats = compute_stats(cache, [0], cfg)
        ds = RounDWindows(cache, [0], cfg, stats, n_pred=cfg.data.n_pred)
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=cfg.data.n_cond)).eval()
        with torch.no_grad():
            m.P_z.weight.normal_(0, 0.3)                       # make z matter in the untrained model
        ck = os.path.join(root, "runs", "fake", "last.pt")
        os.makedirs(os.path.dirname(ck))
        torch.save({"model": m.state_dict(), "wm_cfg": vars(m.cfg), "stats": stats}, ck)
        items = [ds[i] for i in range(3)]
        b = {k: torch.stack([it[k] for it in items]) for k in ("images", "actions", "speed", "ego_state")}
        _S.update(root=root, cfg=cfg, raw=raw, cache=cache, stats=stats, ds=ds, m=m, ck=ck, b=b, items=items)
    return _S


def _start(S):
    m, b, cfg = S["m"], S["b"], S["cfg"]
    n = cfg.data.n_cond
    with torch.no_grad():
        h0, _ = m.warmup(m.encode(b["images"][:, :n]), b["actions"], b["speed"])
    return h0, b["images"][:, n - 1, GREEN:GREEN + 1], routes_from_batch(b, S["stats"], cfg)


def test_rollout_with_fixed_accels_matches_imagine():
    """Same accelerations, z = prior mean: policy_rollout and the planner's imagine() give the same frames, speeds
    and curvatures, so policy and planner see the same imagined world."""
    S = _setup()
    m, cfg, stats = S["m"], S["cfg"], S["stats"]
    h0, prev0, (v0, rs, rk, acc_prev, logged) = _start(S)
    T = 6
    acc = np.stack([np.linspace(-2, 1.5, T), np.full(T, 0.5), logged[2, :T]])
    out = policy_rollout(m, h0, prev0, v0, rs, rk, stats, cfg, T, accels=acc, eps="mean")
    for bi in range(3):
        one = policy_rollout(m, h0[bi:bi + 1], prev0[bi:bi + 1], v0[bi:bi + 1], rs[bi:bi + 1], rk[bi:bi + 1], stats,
                             cfg, T, accels=acc[bi:bi + 1], eps="mean")
        st0 = S["items"][bi]["ego_state"][cfg.data.n_cond - 1].numpy().astype(np.float64)
        paths = ego_paths(st0, rs[bi], rk[bi], acc[bi:bi + 1], cfg)
        ref = imagine(m, h0[bi:bi + 1], paths, stats, cfg, n_samples=1, n_masks=1, prev0=prev0[bi:bi + 1],
                      z_mode="mean")
        assert np.allclose(out["v"][bi].numpy(), paths["state"][0, :, 3], atol=1e-4)
        assert np.allclose(out["curv"][bi].numpy(), paths["curv"][0], atol=1e-6)
        assert torch.allclose(one["green"][0], ref["green"][0, 0, 0], atol=1e-6)        # alone: identical
        # in a batch, batch-size-dependent convolution numerics differ slightly (amplified by the untrained model)
        assert torch.allclose(out["green"][bi], ref["green"][0, 0, 0], atol=5e-3)


def test_policy_range_start_and_gradient():
    import copy
    S = _setup()
    m, cfg, stats = copy.deepcopy(S["m"]).requires_grad_(False), S["cfg"], S["stats"]    # frozen, as in training
    h0, prev0, (v0, rs, rk, acc_prev, _) = _start(S)
    pi = PolicyNet(hidden=m.cfg.hidden)
    out = policy_rollout(m, h0, prev0, v0, rs, rk, stats, cfg, 4, policy=pi)
    assert torch.allclose(out["acc"], torch.zeros_like(out["acc"]), atol=1e-5)          # starts as coast
    c = rollout_cost(out, cfg, CostConfig(w_prog=0.5), acc_prev)
    c["total"].mean().backward()
    assert pi.mlp[-1].weight.grad.abs().sum() > 0                                       # the cost reaches the policy
    assert all(p.grad is None for p in m.parameters())                                  # the world model is frozen
    with torch.no_grad():
        pi.mlp[-1].bias.fill_(50.0)
        hi = pi(h0, torch.zeros(3), torch.zeros(3, 4))
        pi.mlp[-1].bias.fill_(-50.0)
        lo = pi(h0, torch.zeros(3), torch.zeros(3, 4))
    assert torch.allclose(hi, torch.full_like(hi, A_HI)) and torch.allclose(lo, torch.full_like(lo, A_LO))


def test_uncertainty_zero_without_dropout_and_positive_with_it():
    S = _setup()
    cfg, stats = S["cfg"], S["stats"]
    h0, prev0, (v0, rs, rk, _, logged) = _start(S)
    for p, pos in ((0.0, False), (0.1, True)):
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=cfg.data.n_cond, dropout=p)).eval()
        m.load_state_dict(S["m"].state_dict())
        with torch.no_grad():
            out = policy_rollout(m, h0, prev0, v0, rs, rk, stats, cfg, 3, accels=logged[:, :3], n_masks=3)
        assert out["unc"].shape == (3, 3) and ((out["unc"] > 1e-8).all() if pos else (out["unc"].abs() < 1e-8).all())
        assert not m.training


def test_train_then_replay_smoke():
    S = _setup()
    out_root = os.path.join(S["root"], "pol")
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_policy.py"), "--cache", S["cache"],
                        "--wm", S["ck"], "--name", "pol_fake", "--out_root", out_root, "--steps", "4", "--batch", "4",
                        "--T", "3", "--masks", "2", "--lam", "0.5", "--calib_windows", "8", "--n_val", "6",
                        "--val_every", "2", "--log_every", "2", "--workers", "0"],
                       capture_output=True, text=True, env=env, cwd=S["root"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "uncertainty normalisation" in r.stdout and "VAL step 4" in r.stdout
    ck = os.path.join(out_root, "pol_fake", "best.pt")
    assert os.path.exists(ck) and os.path.exists(os.path.join(out_root, "pol_fake", "last.pt"))
    # the trained policy drives in replay, directly and through replay_eval
    ps = torch.load(ck)
    pi = PolicyNet(hidden=S["m"].cfg.hidden)
    pi.load_state_dict(ps["policy"])
    renv = ReplayEnv(S["cache"], S["raw"], [0], S["cfg"], stats=S["stats"], horizon=5)
    pol = LearnedPolicy(S["m"], pi, S["stats"], S["cfg"])
    met = run_episode(renv, pol, fake_round.ego_window(renv))
    summ = pol.episode_summary()
    assert met["steps"] > 0 and A_LO <= summ["pol_acc_mean"] <= A_HI
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_eval.py"), "--cache", S["cache"],
                         "--data_dir", S["raw"], "--n_windows", "2", "--horizon", "4", "--policies", "human",
                         "--policy_ckpts", ck, "--out", os.path.join(S["root"], "re.csv")],
                        capture_output=True, text=True, env=env, cwd=S["root"])
    assert r2.returncode == 0, r2.stdout[-1500:] + r2.stderr[-2500:]
    assert "pol:pol_fake" in r2.stdout
