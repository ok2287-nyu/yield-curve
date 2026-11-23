"""Closed-loop model-predictive planner for ReplayEnv (controller A of the policy design, Nov 22).

Every control step (0.2 s):
    1. warm the world model on the last n_cond REAL frames the ego has seen (the window's warm-up at the start, then
       the frames ReplayEnv renders along the ego's own driven path), with the actions and speeds it really used;
    2. imagine P constant-acceleration plans for T steps along the human's route (imagine.imagine: S prior samples x
       M dropout masks; or warp_plan.warp_imagine, the predict-then-warp baseline);
    3. score them with cost.plan_costs (risk averaged over samples, epistemic = std over masks) and apply the first
       acceleration of the cheapest; replan at the next step.

Plans: constant accelerations (default -3, -1.5, 0, +0.75, +1.5 m/s^2) plus "hold" (the acceleration applied at the
previous step). Steering follows the human's route by distance, as in ReplayEnv.

    pol = ModelPlanner(model, stats, cfg, name="mpc:ab_inv4_01")
    metrics = run_episode(env, pol, window); pol.summary()   # pick shares, imagined risk of the picks
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from .cost import CostConfig, plan_costs
from .imagine import ego_paths, imagine
from .warp_plan import warp_imagine

PLAN_ACC = {"brake": -3.0, "gentle": -1.5, "coast": 0.0, "nudge": 0.75, "go": 1.5}


class ModelPlanner:
    def __init__(self, model, stats, cfg, cc: CostConfig | None = None, plans=PLAN_ACC, hold=True, T=10, samples=4,
                 masks=4, seed=0, warp_ref=None, name="mpc", device=None):
        self.m, self.stats, self.cfg = model.eval(), stats, cfg
        self.cc = cc or CostConfig()
        self.plans, self.hold, self.T, self.S, self.M = dict(plans), hold, T, samples, masks
        self.seed, self.warp_ref, self.name = seed, warp_ref, name
        self.dev = device or next(model.parameters()).device
        self.names = list(self.plans) + (["hold"] if hold else [])
        if warp_ref and "coast" not in self.names:
            raise ValueError("warp_ref needs a 'coast' plan")
        self.a_mean = np.asarray(stats["action_mean"], dtype=np.float64)
        self.a_std = np.asarray(stats["action_std"], dtype=np.float64)
        self.picks, self.risks = Counter(), []

    # ---- buffers of what the ego has really seen and done --------------------------------------------------------
    def reset(self, env):
        n = self.cfg.data.n_cond
        it = env.item
        self.frames = [it["images"][i].float() for i in range(n)]                      # (3, 64, 64) in [0, 1]
        self.actions = [it["actions"][i].float() for i in range(n - 1)]                # normalised (acc, curv)
        self.speeds = [it["speed"][i].float() for i in range(n)]                       # normalised (1,)
        self.acc_prev = float(it["actions"][n - 2, 0] * self.a_std[0] + self.a_mean[0])
        rs, rk = env.path.route_table(self.cfg.data.v_min_curvature, self.cfg.data.curvature_clip)
        self.route_s, self.route_k = rs, rk
        self.window = env.window

    def _observe(self, obs):
        """Append the frame, action and speed of the step just taken (t > 0)."""
        acc, curv = (float(x) for x in obs["action"])
        a = (np.array([acc, curv]) - self.a_mean) / self.a_std
        v = (float(obs["state"][3]) - self.stats["speed_mean"]) / self.stats["speed_std"]
        self.frames.append(torch.from_numpy(np.asarray(obs["image"])).float().div(255.0))
        self.actions.append(torch.tensor(a, dtype=torch.float32))
        self.speeds.append(torch.tensor([v], dtype=torch.float32))
        n = self.cfg.data.n_cond
        self.frames, self.actions, self.speeds = self.frames[-n:], self.actions[-(n - 1):], self.speeds[-n:]

    @torch.no_grad()
    def warm(self):
        """-> h (1, C, 8, 8), prev green (1, 1, 64, 64) from the last n real frames."""
        imgs = torch.stack(self.frames)[None].to(self.dev)
        act = torch.stack(self.actions)[None].to(self.dev)
        spd = torch.stack(self.speeds)[None].to(self.dev)
        h, _ = self.m.warmup(self.m.encode(imgs), act, spd)
        return h, imgs[:, -1, 1:2]

    def accels(self):
        acc = [np.full(self.T, v) for v in self.plans.values()]
        if self.hold:
            acc.append(np.full(self.T, self.acc_prev))
        return np.stack(acc)

    @torch.no_grad()
    def score(self, obs):
        """-> (paths, cost dict) for every plan from the current state."""
        h, prev = self.warm()
        state0 = np.asarray(obs["state"], dtype=np.float64)
        paths = ego_paths(state0, self.route_s - float(obs["s"]), self.route_k, self.accels(), self.cfg)
        seed = self.seed + 1000 * int(self.window) + int(obs["t"])
        if self.warp_ref:
            out = warp_imagine(self.m, h, paths, self.stats, self.cfg, self.S, self.M, seed=seed, prev0=prev,
                               ref=self.warp_ref, names=tuple(self.names))
        else:
            out = imagine(self.m, h, paths, self.stats, self.cfg, self.S, self.M, seed=seed, prev0=prev)
        cm = plan_costs(out["green"], out["speed"], out["acc"], out["curv"], self.cfg.render, self.cc,
                        acc_prev=self.acc_prev)
        return paths, cm

    def act(self, env, obs):
        if obs["t"] > 0:
            self._observe(obs)
        paths, cm = self.score(obs)
        best = int(torch.argmin(cm["total"]))
        a = float(paths["acc"][best, 0])
        self.picks[self.names[best]] += 1
        self.risks.append(float(cm["risk"][best]))
        self.acc_prev = a
        return a

    def summary(self):
        """Pick shares and the mean imagined risk of the picked plans, accumulated over the episodes run so far."""
        tot = max(sum(self.picks.values()), 1)
        out = {f"pick_{k}": self.picks.get(k, 0) / tot for k in self.names}
        out["imagined_risk"] = float(np.mean(self.risks)) if self.risks else np.nan
        return out

    def episode_summary(self):
        """summary() of the last episode only, then cleared (one row per episode in replay_eval)."""
        out = self.summary()
        self.picks, self.risks = Counter(), []
        return out
