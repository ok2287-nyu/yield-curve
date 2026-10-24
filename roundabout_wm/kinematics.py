"""Ego actions and the kinematic update.

State  s_t = (x, y, heading, speed)        heading in radians, CCW from +x
Action a_t = (acceleration, curvature)     m/s^2 and 1/m

Update over one step dt (midpoint rule, so extraction and integration are exact inverses):
    speed'   = speed + acceleration * dt
    heading' = heading + curvature * speed * dt
    x' = x + dt * (speed + speed')/2 * cos((heading + heading')/2)
    y' = y + dt * (speed + speed')/2 * sin((heading + heading')/2)

Curvature is undefined for a stopped car (division by speed), so below v_min it is set to 0.
The car then cannot turn while (almost) stopped, which is physically right.
The same functions are written for numpy (dataset) and torch (differentiable rollouts).
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Map angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def extract_actions(heading, speed, dt: float, v_min: float, accel_clip: float, curv_clip: float):
    """Actions that move state t to state t+1, for a track sampled every dt.

    heading, speed: arrays of length T. Returns array (T-1, 2).
    """
    heading = np.asarray(heading, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    accel = (speed[1:] - speed[:-1]) / dt
    dtheta = wrap_angle(heading[1:] - heading[:-1])
    moving = speed[:-1] >= v_min
    curv = np.zeros_like(dtheta)
    curv[moving] = dtheta[moving] / (speed[:-1][moving] * dt)
    accel = np.clip(accel, -accel_clip, accel_clip)
    curv = np.clip(curv, -curv_clip, curv_clip)
    return np.stack([accel, curv], axis=1)


def step_np(state, action, dt: float):
    """One kinematic step. state (..., 4) = x, y, heading, speed; action (..., 2)."""
    x, y, th, v = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    acc, curv = action[..., 0], action[..., 1]
    v2 = np.maximum(v + acc * dt, 0.0)          # no reversing
    th2 = th + curv * v * dt
    v_mid, th_mid = 0.5 * (v + v2), 0.5 * (th + th2)
    x2 = x + dt * v_mid * np.cos(th_mid)
    y2 = y + dt * v_mid * np.sin(th_mid)
    return np.stack([x2, y2, wrap_angle(th2), v2], axis=-1)


def rollout_np(state0, actions, dt: float):
    """Integrate a sequence of actions from state0. Returns (T+1, 4) including state0."""
    states = [np.asarray(state0, dtype=np.float64)]
    for a in actions:
        states.append(step_np(states[-1], np.asarray(a), dt))
    return np.stack(states)


def step_torch(state, action, dt: float):
    """Differentiable version of step_np (gradients flow to the action)."""
    import torch
    x, y, th, v = state.unbind(-1)
    acc, curv = action.unbind(-1)
    v2 = torch.clamp(v + acc * dt, min=0.0)
    th2 = th + curv * v * dt
    v_mid, th_mid = 0.5 * (v + v2), 0.5 * (th + th2)
    x2 = x + dt * v_mid * torch.cos(th_mid)
    y2 = y + dt * v_mid * torch.sin(th_mid)
    return torch.stack([x2, y2, th2, v2], dim=-1)  # heading not wrapped: keeps the gradient smooth
