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
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_eval.py"), "--cache", S["cache"],
                         "--data_dir", S["raw"], "--n_windows", "2", "--horizon", "3", "--policies", "human",
                         "--ckpts", S["ck"], "--planners", "mpc", "--samples", "1", "--masks", "2", "--w_epi", "10",
                         "--epi_thresh", "0.1", "--out", out], capture_output=True, text=True,
                        env=dict(os.environ, PYTHONPATH=ROOT), cwd=S["root"])
    assert r2.returncode == 0, r2.stdout[-1500:] + r2.stderr[-2500:]
    assert "mpc:fake@w_epi=10,epi_thresh=0.1" in r2.stdout and "'epi_thresh': 0.1" in r2.stdout


def test_fallback_takes_the_lowest_risk_plan_when_the_pick_is_uncertain(monkeypatch):
    """Costs replaced: 'go' is cheapest but uncertain (epistemic 0.5), 'brake' has the lowest risk. No fallback ->
    go; fallback 0.1 -> brake (counted); fallback 1.0 -> go."""
    S = _setup()
    env, cfg = S["env"], S["cfg"]
    names = ["brake", "gentle", "coast", "nudge", "go", "hold"]

    def fake_imagine(m, h, paths, *a, **k):
        P, T = paths["acc"].shape
        return {"green": torch.zeros(1, 1, P, T, 64, 64), "speed": paths["state"][:, :, 3], "acc": paths["acc"],
                "curv": paths["curv"]}

    def fake_costs(*a, **k):
        return {"total": torch.tensor([5., 4, 3, 2, 1, 6]), "risk": torch.tensor([0.1, 1, 1, 1, 1, 1]),
                "epistemic": torch.tensor([0., 0, 0, 0, 0.5, 0])}

    monkeypatch.setattr(planner_mod, "imagine", fake_imagine)
    monkeypatch.setattr(planner_mod, "plan_costs", fake_costs)
    for fb, want in ((None, "go"), (0.1, "brake"), (1.0, "go")):
        pol = ModelPlanner(S["m"], S["stats"], cfg, samples=1, masks=1, fallback=fb)
        pol.record_trace = True
        obs = env.reset(S["w"])
        pol.reset(env)
        pol.act(env, obs)
        assert pol.trace[-1]["pick"] == want and pol.names == names
        assert pol.trace[-1]["epi_go"] == 0.5 and abs(pol.trace[-1]["risk_brake"] - 0.1) < 1e-6
        summ = pol.episode_summary()
        assert ("fallback_share" in summ) == (fb is not None)
        if fb is not None:
            assert summ["fallback_share"] == (1.0 if want == "brake" else 0.0)


def test_fallback_on_risk_is_the_control(monkeypatch):
    """fallback_on="risk": the trigger is the cheapest plan's imagined RISK (go: 1.0), not its epistemic term.
    Threshold 0.5 -> brake (lowest risk); 2.0 -> go. The epistemic term no longer matters (set to 0 here)."""
    S = _setup()
    env, cfg = S["env"], S["cfg"]

    def fake_imagine(m, h, paths, *a, **k):
        P, T = paths["acc"].shape
        return {"green": torch.zeros(1, 1, P, T, 64, 64), "speed": paths["state"][:, :, 3], "acc": paths["acc"],
                "curv": paths["curv"]}

    def fake_costs(*a, **k):
        return {"total": torch.tensor([5., 4, 3, 2, 1, 6]), "risk": torch.tensor([0.1, 1, 1, 1, 1, 1]),
                "epistemic": torch.zeros(6)}

    monkeypatch.setattr(planner_mod, "imagine", fake_imagine)
    monkeypatch.setattr(planner_mod, "plan_costs", fake_costs)
    for fb, want in ((0.5, "brake"), (2.0, "go")):
        pol = ModelPlanner(S["m"], S["stats"], cfg, samples=1, masks=1, fallback=fb, fallback_on="risk")
        pol.record_trace = True
        obs = env.reset(S["w"])
        pol.reset(env)
        pol.act(env, obs)
        assert pol.trace[-1]["pick"] == want
    epi = ModelPlanner(S["m"], S["stats"], cfg, samples=1, masks=1, fallback=0.5)    # default trigger: epistemic 0
    epi.record_trace = True
    obs = env.reset(S["w"])
    epi.reset(env)
    epi.act(env, obs)
    assert epi.trace[-1]["pick"] == "go"
    try:
        ModelPlanner(S["m"], S["stats"], cfg, fallback_on="gap")
        raise AssertionError("bad fallback_on accepted")
    except ValueError:
        pass


def test_replay_eval_risk_fallback_and_headway_flags():
    S = _setup()
    out = os.path.join(S["root"], "re_ctl.csv")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_eval.py"), "--cache", S["cache"],
                        "--data_dir", S["raw"], "--n_windows", "2", "--horizon", "4", "--policies", "human",
                        "--ckpts", S["ck"], "--planners", "mpc", "--samples", "1", "--masks", "2",
                        "--fallback_thresh", "0.5", "--fallback_on", "risk", "--headway", "--out", out],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=S["root"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "mpc:fake@fallback_risk=0.5+hw" in r.stdout and "headway guard" in r.stdout
    import pandas as pd
    df = pd.read_csv(out)
    g = df[df.policy == "mpc:fake@fallback_risk=0.5+hw"]
    assert len(g) == 2 and g.hw_share.between(0, 1).all() and g.fallback_share.between(0, 1).all()
