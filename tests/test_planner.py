"""controller/planner.py: the closed-loop model planner's buffers, its choice logic, and an end-to-end run."""
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
from roundabout_wm.controller import planner as planner_mod
from roundabout_wm.controller.imagine import warm_states
from roundabout_wm.controller.planner import ModelPlanner
from roundabout_wm.controller.replay import ReplayEnv, run_episode
from roundabout_wm.dataset import compute_stats
from roundabout_wm.models.world_model import WMConfig, WorldModel

_S = {}


def _setup():
    if not _S:
        root = tempfile.mkdtemp()
        cfg, raw, cache = fake_round.make(root)
        stats = compute_stats(cache, [0], cfg)
        env = ReplayEnv(cache, raw, [0], cfg, stats=stats, horizon=8)
        torch.manual_seed(0)
        m = WorldModel(WMConfig(n_cond=cfg.data.n_cond)).eval()
        ck = os.path.join(root, "runs", "fake", "last.pt")
        os.makedirs(os.path.dirname(ck))
        torch.save({"model": m.state_dict(), "wm_cfg": vars(m.cfg), "stats": stats}, ck)
        _S.update(root=root, cfg=cfg, raw=raw, cache=cache, stats=stats, env=env, m=m, ck=ck,
                  w=fake_round.ego_window(env))
    return _S


def test_warm_start_matches_the_window_and_buffers_follow_the_drive():
    """At t = 0 the planner's state is the cached window's warm-up; after steps, its frames are the frames the
    environment rendered, its actions the ones applied, and the buffers keep n_cond / n_cond - 1 entries."""
    S = _setup()
    env, cfg = S["env"], S["cfg"]
    n = cfg.data.n_cond
    pol = ModelPlanner(S["m"], S["stats"], cfg, samples=1, masks=1)
    obs = env.reset(S["w"])
    pol.reset(env)
    h0, _ = pol.warm()
    h_ref, _ = warm_states(S["m"], [env.item], cfg, "cpu")
    assert torch.allclose(h0, h_ref, atol=1e-6)
    seen = []
    for _ in range(3):
        a = pol.act(env, obs)
        obs = env.step(a)
        seen.append((obs["image"].copy(), a))
    pol._observe(obs)
    assert len(pol.frames) == n and len(pol.actions) == n - 1 and len(pol.speeds) == n
    for k, (img, a) in enumerate(seen):
        j = n - 3 + k
        assert torch.equal(pol.frames[j], torch.from_numpy(img).float().div(255.0))
        acc = float(pol.actions[j - 1][0]) * pol.a_std[0] + pol.a_mean[0]
        assert abs(acc - a) < 1e-4


def test_choice_logic_brakes_for_a_car_ahead_and_goes_on_an_empty_road(monkeypatch):
    """With imagination replaced: a road user drawn on the ego's footprint whenever a plan accelerates -> brake;
    an empty road -> the fastest plan (progress). Comfort weights off: with the defaults (w_acc 0.02) coasting at
    6 m/s on an empty road beats +1.5 m/s^2 (comfort 0.45 > progress gained ~0.25), a calibration question."""
    from roundabout_wm.controller.cost import CostConfig
    S = _setup()
    env, cfg = S["env"], S["cfg"]
    pol = ModelPlanner(S["m"], S["stats"], cfg, cc=CostConfig(w_acc=0.0, w_jerk=0.0), samples=1, masks=1)

    def fake(blocked):
        def imagine(m, h, paths, stats, c, S_, M_, seed=0, prev0=None, **kw):
            P, T = paths["acc"].shape
            g = torch.zeros(1, 1, P, T, 64, 64)
            if blocked:
                ahead = torch.as_tensor(paths["acc"] > 0.1)
                g[0, 0, ahead] = 1.0                                 # the whole frame green for accelerating plans
            return {"green": g, "speed": paths["state"][:, :, 3], "acc": paths["acc"], "curv": paths["curv"]}
        return imagine

    for blocked, want in ((True, ("brake", "gentle", "coast")), (False, ("go",))):
        monkeypatch.setattr(planner_mod, "imagine", fake(blocked))
        obs = env.reset(S["w"])
        pol.reset(env)
        pol.picks.clear()
        a = pol.act(env, obs)
        name = next(iter(pol.picks))
        assert name in want, (blocked, name, a)
        assert (a <= 0.0) if blocked else (a == 1.5)


def test_episode_and_replay_eval_smoke():
    S = _setup()
    env, cfg = S["env"], S["cfg"]
    for warp in (None, "coast"):
        pol = ModelPlanner(S["m"], S["stats"], cfg, samples=2, masks=2, warp_ref=warp)
        met = run_episode(env, pol, S["w"])
        summ = pol.episode_summary()
        assert met["steps"] > 0 and abs(sum(v for k, v in summ.items() if k.startswith("pick_")) - 1) < 1e-6
        assert np.isfinite(summ["imagined_risk"])
    out = os.path.join(S["root"], "re.csv")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_eval.py"), "--cache", S["cache"],
                        "--data_dir", S["raw"], "--n_windows", "3", "--horizon", "5", "--policies", "human,coast",
                        "--ckpts", S["ck"], "--planners", "mpc,mpc_warp", "--samples", "1", "--masks", "2",
                        "--out", out], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT),
                       cwd=S["root"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "mpc:fake" in r.stdout and "mpc_warp:fake" in r.stdout and "imagined risk" in r.stdout
