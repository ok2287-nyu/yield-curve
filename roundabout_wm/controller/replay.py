"""Log-replay environment: closed-loop evaluation of acceleration controllers among the logged road users of rounD.

    env = ReplayEnv(cache_dir, data_dir, rec_ids, cfg, stats, horizon=30)
    obs = env.reset(window)            # the window's warm-up (cached frames 0..n_cond-1) has already happened
    while not env.done:
        obs = env.step(accel)          # m/s^2; env.step(None) takes the human's exact next logged pose
    m = env.metrics()

The ego drives along the human's own path (the logged positions of the ego track from the last warm-up frame on),
by distance travelled; the controller chooses only the acceleration (steering = the human's route, as in the
planner). Every other road user is replayed from the log, so nobody reacts to the ego: the known limitation of log
replay, which the collision classes below keep visible. Frames are rendered exactly as the training cache was
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


class ReplayEnv:
    def __init__(self, cache_dir, data_dir, rec_ids, cfg, stats=None, horizon=30, road=True, recordings=None,
                 stop_on_collision=True):
        """horizon: control steps per episode (30 = 6 s) after the warm-up; shorter if the recording ends.
        stop_on_collision: end the episode at the first overlap (after it, a replayed car passes through the ego and
        would be counted again, in another class).
        recordings: optional {rec_id: object like _Recording} (tests)."""
        self.cfg, self.cache_dir, self.data_dir, self.rec_ids = cfg, cache_dir, data_dir, list(rec_ids)
        self.ds = RounDWindows(cache_dir, self.rec_ids, cfg, stats, n_pred=cfg.data.n_pred)
        self.horizon, self.use_road, self.stop_on_collision = horizon, road, stop_on_collision
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
        self._record(acc=np.nan, curv=np.nan)
        return self.obs()

    def ego_dict(self, pose=None):
        x, y, h = pose or self.pose
        return dict(x=x, y=y, heading=h, length=self.ego_size[0], width=self.ego_size[1])

    def frame_at(self, t):
        return self.f0 + t * self.cfg.data.step

    def render(self, pose=None, t=None):
        t = self.t if t is None else t
        road = self.R.road if self.use_road else None
        return render_frame(self.ego_dict(pose), self.R.others(self.frame_at(t), self.ego_id), road, self.cfg.render)

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
        others = self.R.others(self.frame_at(self.t), self.ego_id)
        ego = self.ego_dict()
        self.log.append(dict(t=self.t, s=self.s, v=self.v, acc=acc, curv=curv, x=self.pose[0], y=self.pose[1],
                             heading=self.pose[2], gap=front_gap(ego, others), hits=collisions(ego, others)))

    def open_loop(self, speeds):
        """Open-loop plans from the episode start (call right after reset). speeds (P, T + 1) m/s along the path.
        -> green uint8 (P, T, 64, 64) real frames at the plans' poses with the logged road users;
           green_nf the same without followers (road users whose centre is more than a quarter of the ego's length
           behind the ego's centre: a plan that slows down is not charged for the logged follower driving into it);
           front, side, rear (P, T) bool collisions; hit_ids[p][t] list of (trackId, class); gap (P, T) metres.
        Steps past the end of the recording are dropped (T shrinks)."""
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
        for t in range(T):
            others = self.R.others(self.frame_at(t + 1), self.ego_id)
            for p in range(P):
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
        return {"green": green, "green_nf": green_nf, "poses": poses, "gap": gap, "hit_ids": hit_ids, **hits}

    def footprint(self, track_id, pose, t):
        """(64, 64) float in [0, 1]: road user `track_id` alone, at frame t of the episode, seen from ego pose."""
        o = self.R.others(self.frame_at(t), self.ego_id)
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
