"""Headway guard: a physics safety layer over any ReplayEnv controller (planner or learned policy).

Why. The world model is trusted to 2 s and cannot draw a collision the ego causes, so a controller planning in it
sees an impending rear-end too late: braking from 8 m/s at 3 m/s^2 takes ~2.7 s. The guard looks further with the
simplest model there is, using only what is OBSERVED now (what a radar would give):
    gap      the real gap from the ego's front to the nearest road user in its lane corridor ahead (env.log[-1]["gap"])
    v_lead   that road user's speed along the path, from how the gap changed over the last step:
             v_lead = mean ego speed over the step + (gap_now - gap_prev) / dt   (smoothed, floored at 0)
and assumes the lead keeps that speed. For a proposed acceleration a (held for plan_T steps, then 0), it projects
the gap over H steps and computes the margin  min_t [gap_t - (d0 + tau * v_ego_t)],  t = 1 .. H.
If the proposal's margin is negative it applies the largest candidate acceleration below it whose margin is not,
or, if none is safe, the candidate with the largest margin (normally the hardest brake).

No log future, no model: the guard only reads the current and previous real gaps and speeds.
Defaults d0 = 1 m, tau = 0.5 s: the Nov 13 oracle rule with real gaps (recall 1.00, 7 % false alarms on the
human's own plans); H = 20 steps (4 s).
"""
from __future__ import annotations

import numpy as np

from .replay import front_gap

CANDIDATES = (0.75, 0.0, -1.5, -3.0)          # fall-back accelerations, m/s^2 (the planner's plans below "go")


def headway_margin(gap, v_lead, v0, accs, dt=0.2, plan_T=10, H=20, d0=1.0, tau=0.5):
    """gap (m, may be inf), v_lead, v0 (m/s), accs (P,) constant accelerations held for plan_T steps, then 0.
    -> (P,) margin min_{t=1..H} [gap_t - (d0 + tau * v_t)]; inf when there is no road user ahead."""
    accs = np.atleast_1d(np.asarray(accs, dtype=np.float64))
    if not np.isfinite(gap):
        return np.full(accs.shape, np.inf)
    v = np.full(accs.shape, float(v0))
    g = np.full(accs.shape, float(gap))
    best = np.full(accs.shape, np.inf)
    for t in range(1, H + 1):
        a = accs if t <= plan_T else 0.0
        v1 = np.maximum(v + a * dt, 0.0)
        g = g + (v_lead - 0.5 * (v + v1)) * dt
        v = v1
        best = np.minimum(best, g - (d0 + tau * v))
    return best


class LeadTracker:
    """The observed gap ahead and the lead's speed, updated once per control step from the environment's log."""

    def __init__(self, jump=3.0, smooth=0.5):
        self.jump, self.smooth = jump, smooth

    def reset(self, env):
        """Previous gap / speed = the human's logged pose one step before the episode starts (the warm-up's end)."""
        self.dt = env.cfg.data.dt
        self.gap_prev, self.v_prev, self.v_lead = np.inf, env.v, None
        g = env.R.tracks[env.ego_id]
        row = g[g["frame"] == env.f0 - env.cfg.data.step]
        if len(row):
            r = row.iloc[0]
            ego = env.ego_dict((float(r["x"]), float(r["y"]), float(r["heading"])))
            self.gap_prev = front_gap(ego, env.R.others(env.f0 - env.cfg.data.step, env.ego_id))
            self.v_prev = float(r["speed"]) if "speed" in r else env.v

    def update(self, env):
        """-> (gap now, estimated lead speed)."""
        gap, v = float(env.log[-1]["gap"]), float(env.v)
        if np.isfinite(gap) and np.isfinite(self.gap_prev) and abs(gap - self.gap_prev) < self.jump + v * self.dt:
            est = max(0.5 * (v + self.v_prev) + (gap - self.gap_prev) / self.dt, 0.0)
            self.v_lead = est if self.v_lead is None else self.smooth * self.v_lead + (1 - self.smooth) * est
        else:                                      # new lead (or none): no closing speed known yet -> same speed
            self.v_lead = v if np.isfinite(gap) else None
        self.gap_prev, self.v_prev = gap, v
        return gap, (self.v_lead if self.v_lead is not None else v)


class HeadwayGuard:
    """Wraps a controller: base.act proposes, the guard may replace the acceleration (see the module docstring)."""

    def __init__(self, base, d0=1.0, tau=0.5, H=20, plan_T=10, candidates=CANDIDATES, name=None):
        self.base, self.d0, self.tau, self.H, self.plan_T = base, d0, tau, H, plan_T
        self.candidates = tuple(sorted(candidates, reverse=True))
        self.name = name or f"{getattr(base, 'name', 'ctrl')}+hw"
        self.lead = LeadTracker()
        self.n_steps = self.n_override = 0
        self.trace = []
        self.record_trace = False

    def reset(self, env):
        self.base.reset(env)
        self.lead.reset(env)

    def act(self, env, obs):
        a = self.base.act(env, obs)                # always: the base keeps its own buffers up to date
        gap, v_lead = self.lead.update(env)
        if a is None:                              # human_exact-style controllers: nothing to guard
            return a
        dt = env.cfg.data.dt
        m = headway_margin(gap, v_lead, env.v, [a], dt, self.plan_T, self.H, self.d0, self.tau)[0]
        out, over = a, False
        if m < 0:
            cands = [c for c in self.candidates if c < a] or [min(self.candidates)]
            ms = headway_margin(gap, v_lead, env.v, cands, dt, self.plan_T, self.H, self.d0, self.tau)
            safe = [c for c, x in zip(cands, ms) if x >= 0]
            out, over = (safe[0] if safe else cands[int(np.argmax(ms))]), True
            if hasattr(self.base, "acc_prev"):    # the planner's "hold" plan should hold what was applied
                self.base.acc_prev = out
        self.n_steps += 1
        self.n_override += int(over)
        if self.record_trace:
            self.trace.append(dict(t=int(obs["t"]), gap=gap, v_lead=v_lead, proposed=a, applied=out, margin=m))
        return out

    def summary(self):
        out = self.base.summary() if hasattr(self.base, "summary") else {}
        out["hw_share"] = self.n_override / max(self.n_steps, 1)
        return out

    def episode_summary(self):
        out = self.base.episode_summary() if hasattr(self.base, "episode_summary") else {}
        out["hw_share"] = self.n_override / max(self.n_steps, 1)
        self.n_steps = self.n_override = 0
        return out
