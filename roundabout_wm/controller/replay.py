"""Log-replay environment: closed-loop evaluation of acceleration controllers among the logged road users of rounD.

    env = ReplayEnv(cache_dir, data_dir, rec_ids, cfg, stats, horizon=30)
    obs = env.reset(window)            # the window's warm-up (cached frames 0..n_cond-1) has already happened
    while not env.done:
        obs = env.step(accel)          # m/s^2; env.step(None) takes the human's exact next logged pose
    m = env.metrics()

The ego drives along the human's own path (the logged positions of the ego track from the last warm-up frame on),
by distance travelled; the controller chooses only the acceleration (steering = the human's route, as in the
planner). Every other road user is replayed from the log, so nobody reacts to the ego: the known limitation of log
replay, which the collision classes below keep visible.

reactive=True (off by default) is the opposite bound: every road user except pedestrians keeps its own logged path
and its logged timing, but brakes (IDM interaction term) whenever the ego, or a road user the ego has already
delayed, is closer ahead in its lane than in the log. ReactiveSim below. Under the human's own plan nothing is
closer than in the log, so the replay stays exact; it only ever slows road users down (they then catch up with their
log at no more than a_max). Replay = nobody yields to the ego; reactive = everybody brakes for it once it is in their
path. Real drivers are in between. Frames are rendered exactly as the training cache was
(render.render_frame with the recording's road map), so at the human's own pose a frame equals the cached one.

Collisions: oriented-rectangle overlap (separating axes) of the ego's box with every road user's box, classified by
where the other road user's centre is in the ego's frame:
    front  ahead of the ego's centre by more than a quarter of the ego's length: the ego drove into it (at fault)
    rear   behind by more than a quarter length: rear-ended; in log replay mostly an artefact (the logged follower
           cannot know that this ego is slower than the human was)
    side   alongside: a cut-in or side swipe (where conflicts at the yield line show up)
Other metrics: minimum gap ahead in the ego's lane corridor, hard braking, progress against the human, freezing
(standing still where the human was moving).
Rendered rounD data: run on HPC / your own machine only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..dataset import RounDWindows
from ..kinematics import wrap_angle
from ..render import RoadMap, car_corners, render_frame
from ..round_io import load_recording

HARD_BRAKE = 3.0      # m/s^2: braking harder than this counts as a hard brake
FREEZE_V = 0.5        # m/s: the ego counts as standing below this
FREEZE_HUMAN_V = 2.0  # m/s: ... where the human, at the same place on the path, was faster than this
CORRIDOR = 0.5        # m beyond the ego's half width that still counts as "in the ego's lane" for the front gap


# ---- geometry ----------------------------------------------------------------------------------------------------
def rects_overlap(a, b):
    """a (4, 2) corners of one rectangle; b (N, 4, 2). -> (N,) bool: do the rectangles intersect (separating axes)."""
    b = np.asarray(b, dtype=np.float64)
    if len(b) == 0:
        return np.zeros(0, bool)
    a = np.asarray(a, dtype=np.float64)
    hit = np.ones(len(b), bool)
    axes_a = np.stack([a[1] - a[0], a[3] - a[0]])                                  # (2, 2)
    axes_b = np.stack([b[:, 1] - b[:, 0], b[:, 3] - b[:, 0]], 1)                  # (N, 2, 2)
    axes = [np.broadcast_to(axes_a[k], (len(b), 2)) for k in range(2)] + [axes_b[:, k] for k in range(2)]
    for ax in axes:                                                                 # (N, 2) each
        pa = np.einsum("cd,nd->nc", a, ax)                                          # (N, 4)
        pb = np.einsum("ncd,nd->nc", b, ax)
        hit &= (pa.max(1) >= pb.min(1)) & (pb.max(1) >= pa.min(1))
    return hit


def to_ego(points, x, y, heading):
    """World points (..., 2) -> (forward, left) metres in the ego frame (..., 2)."""
    d = np.asarray(points, dtype=np.float64) - np.array([x, y])
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)


def box_points(corners):
    """(N, 4, 2) corners -> (N, 9, 2): corners, edge midpoints and centre (so a car across the lane is seen)."""
    mid = 0.5 * (corners + np.roll(corners, -1, axis=1))
    return np.concatenate([corners, mid, corners.mean(1, keepdims=True)], 1)


def collision_class(fwd, ego_length):
    """fwd: the other road user's centre, metres ahead of the ego's centre."""
    q = ego_length / 4
    return "front" if fwd > q else ("rear" if fwd < -q else "side")


def front_gap(ego, others):
    """Metres from the ego's front to the nearest part of any road user in its lane corridor ahead (inf if none)."""
    if not len(others["x"]):
        return np.inf
    pts = box_points(car_corners(others["x"], others["y"], others["heading"], others["length"], others["width"]))
    fl = to_ego(pts, ego["x"], ego["y"], ego["heading"])
    front = ego["length"] / 2
    ok = (np.abs(fl[..., 1]) < ego["width"] / 2 + CORRIDOR) & (fl[..., 0] > 0)
    if not ok.any():
        return np.inf
    return float(max(fl[..., 0][ok].min() - front, 0.0))


def collisions(ego, others):
    """-> list of (trackId, class) for every road user overlapping the ego."""
    if not len(others["x"]):
        return []
    oc = car_corners(others["x"], others["y"], others["heading"], others["length"], others["width"])
    ec = car_corners(ego["x"], ego["y"], ego["heading"], ego["length"], ego["width"])[0]
    hit = rects_overlap(ec, oc)
    out = []
    for j in np.flatnonzero(hit):
        fwd = to_ego(np.array([others["x"][j], others["y"][j]]), ego["x"], ego["y"], ego["heading"])[0]
        out.append((int(others["trackId"][j]), collision_class(fwd, ego["length"])))
    return out


# ---- the human's path --------------------------------------------------------------------------------------------
class HumanPath:
    """The logged ego track from the last warm-up frame on, as a path parametrised by arc length.

    xy (L, 2), heading (L,), speed (L,): logged rows at 5 Hz. Arc length from the positions. Where the human stood
    still, several rows share one arc length: the pose there is the first of them (headings barely change)."""

    def __init__(self, xy, heading, speed):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.heading = np.unwrap(np.asarray(heading, dtype=np.float64))
        self.speed = np.asarray(speed, dtype=np.float64)
        seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])            # human's arc length at each logged step
        keep = np.concatenate([[True], np.diff(self.s) > 1e-6])     # strictly increasing knots for interpolation
        self._s, self._xy, self._h, self._v = self.s[keep], self.xy[keep], self.heading[keep], self.speed[keep]
        self.length = float(self.s[-1])

    def pose(self, s):
        s = float(np.clip(s, 0.0, self.length))
        x = np.interp(s, self._s, self._xy[:, 0]); y = np.interp(s, self._s, self._xy[:, 1])
        return x, y, float(wrap_angle(np.interp(s, self._s, self._h)))

    def poses(self, s):
        """s (...,) -> (..., 3) x, y, heading."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        return np.stack([np.interp(s, self._s, self._xy[:, 0]), np.interp(s, self._s, self._xy[:, 1]),
                         wrap_angle(np.interp(s, self._s, self._h))], -1)

    def human_speed_at(self, s):
        """The human's speed when it was at arc length s. Over a stretch where it stood still, the lowest speed."""
        s = np.asarray(s, dtype=np.float64)
        v = np.interp(s, self._s, self._v)
        # rows dropped as duplicates were the stationary ones: at their arc length the human was standing
        stood = self.s[~np.concatenate([[True], np.diff(self.s) > 1e-6])]
        if len(stood):
            near = np.abs(s[..., None] - stood).min(-1) < 0.05
            v = np.where(near, 0.0, v)
        return v

    def time_at(self, s, dt):
        """Seconds the human needed from the start to reach arc length s (first time)."""
        j = np.searchsorted(self.s, s - 1e-9, side="left")
        if j >= len(self.s):
            return np.inf
        if j == 0:
            return 0.0
        f = (s - self.s[j - 1]) / max(self.s[j] - self.s[j - 1], 1e-9)
        return (j - 1 + f) * dt

    def route_table(self, v_min, curv_clip):
        """(route_s, route_k) for imagine.ego_paths: the human's curvature per logged step against arc length,
        computed like kinematics.extract_actions (heading change / distance)."""
        ds = np.diff(self.s)
        dth = np.diff(self.heading)
        k = np.where(ds > 1e-3, dth / np.maximum(ds, 1e-3), 0.0)
        return self.s.copy(), np.clip(k, -curv_clip, curv_clip)


# ---- data -------------------------------------------------------------------------------------------------------
class _Recording:
    def __init__(self, data_dir, rec_id, cfg):
        d = cfg.data
        rec = load_recording(data_dir, rec_id, step=d.step)
        self.meta = rec.tracks_meta
        tr = rec.tracks.join(rec.tracks_meta[["length", "width"]], on="trackId")
        self.frames = {int(f): {k: g[k].to_numpy() for k in ("trackId", "x", "y", "heading", "length", "width")}
                       for f, g in tr.groupby("frame", sort=False)}
        self.tracks = {int(t): g.reset_index(drop=True) for t, g in rec.tracks.groupby("trackId")}
        self.last_frame = int(rec.tracks["frame"].max())
        self.road = RoadMap(rec.background_path, rec.background_m_per_px(d.background_scale_down), cfg.render)

    def others(self, frame, ego_id):
        o = self.frames.get(int(frame))
        if o is None:
            return {k: np.zeros(0) for k in ("trackId", "x", "y", "heading", "length", "width")}
        m = o["trackId"] != ego_id
        return {k: v[m] for k, v in o.items()}


# ---- reactive road users (reactive=True) -------------------------------------------------------------------------
REACT = dict(s0=2.0, t_head=1.0, a_max=2.0, b_comf=2.0, b_max=8.0)   # IDM: m, s, m/s^2 (catch-up), m/s^2, m/s^2
NON_REACTIVE = ("pedestrian",)


def lane_gaps(agent, bodies):
    """agent dict (scalars x, y, heading, length, width); bodies dict of arrays (N,) -> (N,) metres from the agent's
    front to the nearest part of each body inside the agent's lane corridor ahead (inf if none). front_gap per body."""
    if not len(bodies["x"]):
        return np.zeros(0)
    pts = box_points(car_corners(bodies["x"], bodies["y"], bodies["heading"], bodies["length"], bodies["width"]))
    fl = to_ego(pts, agent["x"], agent["y"], agent["heading"])                   # (N, 9, 2)
    ok = (np.abs(fl[..., 1]) < agent["width"] / 2 + CORRIDOR) & (fl[..., 0] > 0)
    d = np.where(ok, fl[..., 0], np.inf).min(1) - agent["length"] / 2
    return np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)


def idm_brake(v, gaps, gaps_log, v_lead, p=REACT):
    """IDM interaction term against each leader, with the desired gap capped at the gap the road user accepted in
    the log (so the logged behaviour itself is never braked). -> the most negative acceleration (<= 0), or 0."""
    a = 0.0
    for g, gl, vl in zip(gaps, gaps_log, v_lead):
        if not np.isfinite(g):
            continue
        s_star = p["s0"] + max(0.0, v * p["t_head"] + v * (v - vl) / (2 * np.sqrt(p["a_max"] * p["b_comf"])))
        s_star = min(s_star, gl)
        a = min(a, p["a_max"] * (1.0 - (s_star / max(g, 0.1)) ** 2))
    return max(a, -p["b_max"])


class _Agent:
    def __init__(self, tid, g, f0, step, length, width, reactive):
        """g: the road user's logged rows inside the episode (frames f0 + k * step, contiguous)."""
        self.tid, self.length, self.width, self.reactive = tid, length, width, reactive
        self.k0 = int((int(g.frame.iloc[0]) - f0) // step)                       # first episode step it exists
        self.raw = g[["x", "y", "heading"]].to_numpy(np.float64)                  # (L, 3)
        self.path = HumanPath(self.raw[:, :2], self.raw[:, 2], g.speed.to_numpy(np.float64))
        self.s_log = self.path.s                                                  # (L,) arc length at each row
        self.s, self.v = 0.0, float(g.speed.iloc[0])

    def alive(self, t):
        return self.k0 <= t < self.k0 + len(self.raw)

    def logged(self, t):
        return self.raw[t - self.k0]

    def on_log(self, t):
        return abs(self.s - self.s_log[t - self.k0]) < 1e-6

    def pose(self, t):
        return tuple(self.logged(t)) if self.on_log(t) else self.path.pose(self.s)


class ReactiveSim:
    """The road users of one episode, stepped together with the ego (one plan). others() = log replay until the ego
    gets closer to someone than it was in the log."""

    def __init__(self, R, ego_id, f0, step, T, ego_log, dt, p=REACT):
        """ego_log (L, 3): the human's logged poses from the episode start (the tolerance reference)."""
        self.p, self.dt, self.t, self.ego_log = p, dt, 0, np.asarray(ego_log, dtype=np.float64)
        cls = R.meta["class"] if "class" in R.meta else None
        self.agents = []
        for tid, g in R.tracks.items():
            if tid == ego_id:
                continue
            g = g[(g.frame >= f0) & (g.frame <= f0 + T * step)]
            if not len(g):
                continue
            c = str(cls.get(tid, "car")) if cls is not None else "car"
            self.agents.append(_Agent(int(tid), g, f0, step, float(R.meta.at[tid, "length"]),
                                      float(R.meta.at[tid, "width"]), c not in NON_REACTIVE))

    def others(self):
        a = [x for x in self.agents if x.alive(self.t)]
        if not a:
            return {k: np.zeros(0) for k in ("trackId", "x", "y", "heading", "length", "width")}
        pose = np.array([x.pose(self.t) for x in a])
        return {"trackId": np.array([x.tid for x in a]), "x": pose[:, 0], "y": pose[:, 1], "heading": pose[:, 2],
                "length": np.array([x.length for x in a]), "width": np.array([x.width for x in a])}

    def delayed(self):
        """trackIds of road users that are off their log (only ever behind it)."""
        return [x.tid for x in self.agents if x.alive(self.t) and not x.on_log(self.t)]

    def advance(self, ego, v_ego):
        """ego dict at step t (x, y, heading, length, width), its speed -> road users at step t + 1."""
        t, p, dt = self.t, self.p, self.dt
        live = [x for x in self.agents if x.alive(t)]
        st = {x.tid: (*x.pose(t), x.v) for x in live}
        off = [x for x in live if not x.on_log(t)]
        el = self.ego_log[t] if t < len(self.ego_log) else None
        new = {}
        for x in live:
            if not x.alive(t + 1):
                continue
            k = t + 1 - x.k0
            s_next = x.s_log[k]
            if not x.reactive:
                new[x.tid] = (s_next, (s_next - x.s) / dt)
                continue
            xs, ys, hs, vs = st[x.tid]
            me = dict(x=xs, y=ys, heading=hs, length=x.length, width=x.width)
            lead = [(ego["x"], ego["y"], ego["heading"], ego["length"], ego["width"], v_ego,
                     tuple(el) if el is not None else None)]
            for o in off:
                if o.tid != x.tid:
                    ox, oy, oh, ov = st[o.tid]
                    lead.append((ox, oy, oh, o.length, o.width, ov, tuple(o.logged(t))))
            arr = np.array([l[:5] for l in lead])
            bodies = dict(x=arr[:, 0], y=arr[:, 1], heading=arr[:, 2], length=arr[:, 3], width=arr[:, 4])
            gaps = lane_gaps(me, bodies)
            # the same leaders at their logged poses, seen from this road user's logged pose: the gap it accepted
            mx, my, mh = x.logged(t)
            me_log = dict(me, x=mx, y=my, heading=mh)
            gl = np.full(len(lead), np.inf)
            for j, l in enumerate(lead):
                if l[6] is not None:
                    gl[j] = lane_gaps(me_log, dict(x=np.array([l[6][0]]), y=np.array([l[6][1]]),
                                                   heading=np.array([l[6][2]]), length=np.array([l[3]]),
                                                   width=np.array([l[4]])))[0]
            vl = [l[5] * np.cos(l[2] - hs) for l in lead]
            v_free = max((s_next - x.s) / dt, 0.0)
            if x.s < x.s_log[t - x.k0] - 1e-6:                    # behind its log: catch up gently
                v_free = min(v_free, vs + p["a_max"] * dt)
            a = idm_brake(vs, gaps, gl, vl, p)
            v_new = min(v_free, max(vs + a * dt, 0.0)) if a < 0 else v_free
            new[x.tid] = (x.s + v_new * dt if v_new < v_free else s_next, v_new)
        for x in live:
            if x.tid in new:
                x.s, x.v = new[x.tid]
        self.t = t + 1
        for x in self.agents:                                    # road users entering the scene start on their log
            if x.k0 == self.t:
                x.s = 0.0


class ReplayEnv:
    def __init__(self, cache_dir, data_dir, rec_ids, cfg, stats=None, horizon=30, road=True, recordings=None,
                 stop_on_collision=True, reactive=False):
        """horizon: control steps per episode (30 = 6 s) after the warm-up; shorter if the recording ends.
        stop_on_collision: end the episode at the first overlap (after it, a replayed car passes through the ego and
        would be counted again, in another class).
        recordings: optional {rec_id: object like _Recording} (tests).
        reactive: road users react to the ego (ReactiveSim); default False = pure log replay."""
        self.cfg, self.cache_dir, self.data_dir, self.rec_ids = cfg, cache_dir, data_dir, list(rec_ids)
        self.ds = RounDWindows(cache_dir, self.rec_ids, cfg, stats, n_pred=cfg.data.n_pred)
        self.horizon, self.use_road, self.stop_on_collision = horizon, road, stop_on_collision
        self.reactive = reactive
        self.sim, self._plan_others = None, None
        self._recs = dict(recordings or {})
        self._tracks_csv = {}

    # ---- loading ----
    def _rec(self, rec_id):
        if rec_id not in self._recs:
            self._recs[rec_id] = _Recording(self.data_dir, rec_id, self.cfg)
        return self._recs[rec_id]

    def _locate(self, window):
        """-> rec_id, ego trackId, row offset of the window's first frame inside the ego's track."""
        k, s_row = (int(x) for x in self.ds.index[window])
        rec_id = self.rec_ids[k]
        if rec_id not in self._tracks_csv:
            self._tracks_csv[rec_id] = pd.read_csv(os.path.join(self.cache_dir, f"rec_{rec_id:02d}", "tracks.csv"))
        t = self._tracks_csv[rec_id]
        row = t[(t.start <= s_row) & (s_row < t.start + t.length)].iloc[0]
        return rec_id, int(row.trackId), s_row - int(row.start)

    # ---- episode ----
    def reset(self, window):
        cfg = self.cfg
        n, dt = cfg.data.n_cond, cfg.data.dt
        self.window = window
        self.item = self.ds[window]
        self.rec_id, self.ego_id, off = self._locate(window)
        self.R = self._rec(self.rec_id)
        g = self.R.tracks[self.ego_id]
        st = self.item["ego_state"].numpy()
        if not np.allclose(g.loc[off + n - 1, ["x", "y"]].to_numpy(float), st[n - 1, :2], atol=1e-3):
            raise RuntimeError(f"window {window}: cached ego pose does not match the raw track")
        tail = g.iloc[off + n - 1:]
        self.frames = tail["frame"].to_numpy()
        self.path = HumanPath(tail[["x", "y"]].to_numpy(), tail["heading"].to_numpy(), tail["speed"].to_numpy())
        self.f0 = int(self.frames[0])
        self.ego_size = (float(self.R.meta.at[self.ego_id, "length"]), float(self.R.meta.at[self.ego_id, "width"]))
        self.T = min(self.horizon, (self.R.last_frame - self.f0) // cfg.data.step)
        self.t, self.s, self.v = 0, 0.0, float(self.path.speed[0])
        x, y, h = self.path.xy[0, 0], self.path.xy[0, 1], float(wrap_angle(self.path.heading[0]))
        self.pose = (x, y, h)
        self.exact = True                 # still exactly on the human's logged trajectory
        self.done, self.reason = self.T <= 0, ("log_end" if self.T <= 0 else None)
        self.log = []
        self.sim = self._new_sim() if self.reactive else None
        self._plan_others = None
        self._record(acc=np.nan, curv=np.nan)
        return self.obs()

    def _new_sim(self):
        ego_log = np.stack([self.path.xy[:, 0], self.path.xy[:, 1], self.path.heading], 1)
        return ReactiveSim(self.R, self.ego_id, self.f0, self.cfg.data.step, self.T, ego_log, self.cfg.data.dt)

    def others_at(self, t, plan=None):
        """Road users at episode step t: the log, or (reactive) the closed-loop simulation's current step, or the
        road users of open_loop plan `plan` (after open_loop; the log when not reactive)."""
        if self.reactive and plan is not None and self._plan_others is not None:
            return self._plan_others[plan][t]
        if self.reactive and plan is None and self.sim is not None and t == self.sim.t:
            return self.sim.others()
        return self.R.others(self.frame_at(t), self.ego_id)

    def ego_dict(self, pose=None):
        x, y, h = pose or self.pose
        return dict(x=x, y=y, heading=h, length=self.ego_size[0], width=self.ego_size[1])

    def frame_at(self, t):
        return self.f0 + t * self.cfg.data.step

    def render(self, pose=None, t=None):
        t = self.t if t is None else t
        road = self.R.road if self.use_road else None
        return render_frame(self.ego_dict(pose), self.others_at(t), road, self.cfg.render)

    def obs(self):
        """image uint8 (3, 64, 64) at the current pose; state x, y, heading, speed; the action (acc, curv) that led
        here (NaN at reset: the warm-up's last action is the item's); t."""
        last = self.log[-1]
        return {"image": self.render(), "state": np.array([*self.pose, self.v]), "t": self.t,
                "action": np.array([last["acc"], last["curv"]]), "s": self.s}

    def step(self, accel=None):
        """accel m/s^2 (clipped to the data range), or None for the human's exact next logged pose."""
        if self.done:
            raise RuntimeError("episode is over")
        d, dt = self.cfg.data, self.cfg.data.dt
        v0, h0 = self.v, self.pose[2]
        if self.sim is not None:
            self.sim.advance(self.ego_dict(), self.v)        # road users react to the ego as it is now (0.2 s delay)
        if accel is None:
            if not self.exact:
                raise RuntimeError("env.step(None) only while the ego is still on the human's trajectory")
            j = self.t + 1
            if j >= len(self.path.s):
                self.done, self.reason = True, "route_end"
                return self.obs()
            self.s, self.v = float(self.path.s[j]), float(self.path.speed[j])
            self.pose = (float(self.path.xy[j, 0]), float(self.path.xy[j, 1]), float(wrap_angle(self.path.heading[j])))
        else:
            self.exact = False
            a = float(np.clip(accel, -d.accel_clip, d.accel_clip))
            v1 = max(v0 + a * dt, 0.0)
            self.s = self.s + 0.5 * (v0 + v1) * dt
            self.v = v1
            if self.s >= self.path.length:
                self.s = self.path.length
                self.done, self.reason = True, "route_end"
            self.pose = self.path.pose(self.s)
        acc = (self.v - v0) / dt
        curv = wrap_angle(self.pose[2] - h0) / (v0 * dt) if v0 >= d.v_min_curvature else 0.0
        self.t += 1
        self._record(acc=float(np.clip(acc, -d.accel_clip, d.accel_clip)),
                     curv=float(np.clip(curv, -d.curvature_clip, d.curvature_clip)))
        if self.stop_on_collision and self.log[-1]["hits"] and not self.done:
            self.done, self.reason = True, "collision"     # replayed road users would drive through the ego
        if self.t >= self.T and not self.done:
            self.done, self.reason = True, "horizon"
        return self.obs()

    def _record(self, acc, curv):
        others = self.others_at(self.t)
        ego = self.ego_dict()
        self.log.append(dict(t=self.t, s=self.s, v=self.v, acc=acc, curv=curv, x=self.pose[0], y=self.pose[1],
                             heading=self.pose[2], gap=front_gap(ego, others), hits=collisions(ego, others)))

    def open_loop(self, speeds):
        """Open-loop plans from the episode start (call right after reset). speeds (P, T + 1) m/s along the path.
        -> green uint8 (P, T, 64, 64) real frames at the plans' poses with the logged road users;
           green_nf the same without followers (road users whose centre is more than a quarter of the ego's length
           behind the ego's centre: a plan that slows down is not charged for the logged follower driving into it);
           front, side, rear (P, T) bool collisions; hit_ids[p][t] list of (trackId, class); gap (P, T) metres.
        Steps past the end of the recording are dropped (T shrinks).
        reactive: every plan gets its own ReactiveSim (road users react to that plan); delayed[p][t] lists the
        trackIds the plan has slowed down (empty lists when not reactive)."""
        dt, step = self.cfg.data.dt, self.cfg.data.step
        speeds = np.asarray(speeds, dtype=np.float64)
        P, T = speeds.shape[0], min(speeds.shape[1] - 1, (self.R.last_frame - self.f0) // step)
        s = np.concatenate([np.zeros((P, 1)), np.cumsum(0.5 * (speeds[:, 1:] + speeds[:, :-1]) * dt, 1)], 1)
        poses = self.path.poses(s)                                                   # (P, T + 1, 3)
        rs = self.cfg.render.size
        green = np.zeros((P, T, rs, rs), np.uint8)
        green_nf = np.zeros((P, T, rs, rs), np.uint8)
        hits = {c: np.zeros((P, T), bool) for c in ("front", "side", "rear")}
        hit_ids = [[[] for _ in range(T)] for _ in range(P)]
        gap = np.full((P, T), np.inf)
        delayed = [[[] for _ in range(T)] for _ in range(P)]
        if self.reactive:
            self._plan_others = [[None] * (T + 1) for _ in range(P)]
            for p in range(P):
                sim = self._new_sim()
                self._plan_others[p][0] = sim.others()
                for t in range(T):
                    sim.advance(self.ego_dict(tuple(poses[p, t])), float(speeds[p, t]))
                    self._plan_others[p][t + 1] = sim.others()
                    delayed[p][t] = sim.delayed()
        for t in range(T):
            for p in range(P):
                others = self.others_at(t + 1, p) if self.reactive else self.R.others(self.frame_at(t + 1), self.ego_id)
                ego = self.ego_dict(tuple(poses[p, t + 1]))
                green[p, t] = render_frame(ego, others, None, self.cfg.render)[1]
                fwd = to_ego(np.stack([others["x"], others["y"]], -1), ego["x"], ego["y"], ego["heading"])[..., 0] \
                    if len(others["x"]) else np.zeros(0)
                keep = fwd >= -ego["length"] / 4
                green_nf[p, t] = render_frame(ego, {k: v[keep] for k, v in others.items()}, None,
                                              self.cfg.render)[1]
                for tid, c in collisions(ego, others):
                    hits[c][p, t] = True
                    hit_ids[p][t].append((tid, c))
                gap[p, t] = front_gap(ego, others)
        return {"green": green, "green_nf": green_nf, "poses": poses, "gap": gap, "hit_ids": hit_ids,
                "delayed": delayed, "others": self._plan_others if self.reactive else None, **hits}

    def use_plans(self, real):
        """After reset, reuse a cached open_loop result of this window: footprint(plan=...) then follows its
        simulated road users again (reset forgets them)."""
        self._plan_others = real.get("others")

    def footprint(self, track_id, pose, t, plan=None):
        """(64, 64) float in [0, 1]: road user `track_id` alone, at frame t of the episode, seen from ego pose.
        plan: with reactive=True, where that road user is under open_loop plan `plan` (else the log)."""
        o = self.others_at(t, plan) if plan is not None else self.R.others(self.frame_at(t), self.ego_id)
        m = o["trackId"] == track_id
        return render_frame(self.ego_dict(tuple(pose)), {k: v[m] for k, v in o.items()}, None,
                            self.cfg.render)[1].astype(np.float32) / 255.0

    # ---- metrics ----
    def metrics(self):
        dt = self.cfg.data.dt
        L = pd.DataFrame(self.log)
        out = dict(window=self.window, rec=self.rec_id, ego=self.ego_id, steps=int(L.t.iloc[-1]),
                   reason=self.reason)
        for cls in ("front", "side", "rear"):
            ts = [r.t for r in L.itertuples() if any(c == cls for _, c in r.hits)]
            out[f"col_{cls}"] = bool(ts)
            out[f"t_{cls}"] = ts[0] * dt if ts else np.nan
        out["min_gap"] = float(L.gap.min())
        acc = L.acc.to_numpy()[1:]
        out["hard_brakes"] = int((acc < -HARD_BRAKE - 1e-6).sum())
        out["min_acc"] = float(np.min(acc)) if len(acc) else np.nan
        out["jerk_rms"] = float(np.sqrt(np.mean(np.diff(acc) ** 2)) / dt) if len(acc) > 1 else np.nan
        t_end = int(L.t.iloc[-1])
        s_h = self.path.s[min(t_end, len(self.path.s) - 1)]
        out["dist"] = float(self.s)
        out["dist_human"] = float(s_h)
        out["progress"] = float(self.s / s_h) if s_h > 1.0 else np.nan
        out["lag_s"] = float(t_end * dt - self.path.time_at(self.s, dt)) if np.isfinite(
            self.path.time_at(self.s, dt)) else np.nan
        # freezing: standing where the human, at that place on the path, was moving
        stand = (L.v.to_numpy() < FREEZE_V) & (self.path.human_speed_at(L.s.to_numpy()) > FREEZE_HUMAN_V)
        run = best = 0
        for x in stand:
            run = run + 1 if x else 0
            best = max(best, run)
        out["freeze_s"] = best * dt
        return out


# ---- simple policies ---------------------------------------------------------------------------------------------
class HumanExact:
    """Takes the human's exact next logged pose (the acceptance test: no at-fault collisions)."""
    name = "human"

    def reset(self, env):
        pass

    def act(self, env, obs):
        return None


class HumanAccel:
    """The human's logged accelerations, integrated along the path (measures the integration drift)."""
    name = "human_acc"

    def reset(self, env):
        v = env.path.speed
        self.acc = np.diff(v) / env.cfg.data.dt

    def act(self, env, obs):
        return float(self.acc[obs["t"]]) if obs["t"] < len(self.acc) else 0.0


class Constant:
    def __init__(self, a, name=None):
        self.a, self.name = float(a), name or f"const{a:+g}"

    def reset(self, env):
        pass

    def act(self, env, obs):
        return self.a


def run_episode(env, policy, window):
    obs = env.reset(window)
    policy.reset(env)
    while not env.done:
        obs = env.step(policy.act(env, obs))
    return env.metrics()
