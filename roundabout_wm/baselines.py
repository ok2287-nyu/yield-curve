"""Kinematic baselines for the other road users: constant velocity (CV) and constant turn rate (CTRV).

Both start from each road user's logged state at the last warm-up frame and assume it keeps going:
    CV     same speed, same heading: a straight line
    CTRV   same speed and same yaw rate (heading change per second): an arc, so ring traffic follows the curve
Only road users the world model could also have seen are used: visible in the crop at any warm-up frame.
"""
from __future__ import annotations

import numpy as np

from .kinematics import wrap_angle


def extrapolate(x, y, heading, speed, yaw_rate, t, turn: bool):
    """Positions/headings after t seconds. Arrays (N,) -> (x_t, y_t, heading_t), each (N,).
    turn=False: constant velocity. turn=True: constant turn rate (straight where |yaw_rate| is ~0)."""
    x, y, h, v, w = (np.asarray(a, dtype=np.float64) for a in (x, y, heading, speed, yaw_rate))
    if not turn:
        return x + v * t * np.cos(h), y + v * t * np.sin(h), h
    h_t = h + w * t
    straight = np.abs(w) < 1e-4
    w_safe = np.where(straight, 1.0, w)
    x_arc = x + v / w_safe * (np.sin(h_t) - np.sin(h))
    y_arc = y - v / w_safe * (np.cos(h_t) - np.cos(h))
    x_t = np.where(straight, x + v * t * np.cos(h), x_arc)
    y_t = np.where(straight, y + v * t * np.sin(h), y_arc)
    return x_t, y_t, h_t


def yaw_rate(heading_now, heading_before, dt):
    """Heading change per second from two logged headings dt seconds apart (wrapped)."""
    return wrap_angle(np.asarray(heading_now) - np.asarray(heading_before)) / dt
