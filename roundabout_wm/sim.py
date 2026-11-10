"""Drive inside the world model: a step-by-step imagined roundabout.

Start from a real validation window: the model warms up on its first 2 s (10 real frames), exactly as in training.
From then on nothing real is shown to the model. Every tick (0.2 s) you choose the ego's action; the ego moves by
the kinematic model, the road and the ego are drawn from the real map at the new pose, and the OTHER road users are
the model's prediction (decoded from its hidden state h).

    sim = ImaginedDrive.load(ckpt, cache_dir, data_dir, split="val")
    sim.reset(window_index)
    frame = sim.step(accel=1.0, curvature=None)      # curvature=None: follow the logged path's curvature
    sim.save("a"); ...; sim.restore("a")            # branch: compare two plans from the same moment

Notes:
  * The model was trained on 20-step (4 s) rollouts. Past step 20 it is outside what it was trained for; expect
    the deterministic model to smear and fade. `beyond_horizon` tells you when.
  * Deterministic models (stage 1) have one future (the mean). Stochastic models sample z from the learned prior
    each tick; `reset(..., seed=s)` / `reseed(s)` gives another sample of the future from the same state.
  * Frames are rendered from rounD data: run this on HPC / your own machine only.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import RounDWindows
from .kinematics import step_np
from .models.world_model import GREEN, WMConfig, WorldModel
from .render import RoadMap


class ImaginedDrive:
    def __init__(self, model: WorldModel, ds: RounDWindows, rec_ids: list[int], stats: dict, data_dir: str,
                 cfg: Config, device: str = "cpu"):
        self.model, self.ds, self.rec_ids, self.stats, self.cfg, self.dev = model, ds, rec_ids, stats, cfg, device
        self.data_dir = data_dir
        self._roads: dict[int, RoadMap] = {}
        self._saved: dict[str, dict] = {}
        self.a_mean = np.array(stats["action_mean"]); self.a_std = np.array(stats["action_std"])

    # ---- construction ---------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt: str, cache_dir: str, data_dir: str, split: str = "val", device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        s = torch.load(ckpt, map_location=device)
        model = WorldModel(WMConfig(**s["wm_cfg"])).to(device)
        model.load_state_dict(s["model"]); model.eval()
        cfg = Config()
        split_ids = json.load(open(os.path.join(cache_dir, "split.json")))[split]
        ds = RounDWindows(cache_dir, split_ids, cfg, s["stats"], n_pred=cfg.data.n_pred)
        print(f"loaded {ckpt} (step {s.get('step')}, {'deterministic' if model.cfg.deterministic else 'stochastic'})"
              f" | {len(ds)} {split} windows")
        return cls(model, ds, split_ids, s["stats"], data_dir, cfg, device)

    def _road(self, rec: int) -> RoadMap:
        if rec not in self._roads:
            meta = pd.read_csv(os.path.join(self.data_dir, f"{rec:02d}_recordingMeta.csv")).iloc[0]
            m_per_px = float(meta["orthoPxToMeter"]) * self.cfg.data.background_scale_down
            self._roads[rec] = RoadMap(os.path.join(self.data_dir, f"{rec:02d}_background.png"), m_per_px,
                                       self.cfg.render)
        return self._roads[rec]

    # ---- episode ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def reset(self, window: int, seed: int = 0):
        """Warm up on the window's first n_cond real frames. Returns the last real frame (3, 64, 64) in [0, 1]."""
        n = self.cfg.data.n_cond
        item = self.ds[window]
        k, _ = self.ds.index[window]
        self.window, self.rec = window, self.rec_ids[int(k)]
        self.item = item
        b = {key: v[None].to(self.dev) for key, v in item.items()}
        e = self.model.encode(b["images"][:, :n])
        self.h, _ = self.model.warmup(e, b["actions"], b["speed"])
        st = item["ego_state"].numpy().astype(np.float64)
        self.state = st[n - 1].copy()                              # x, y, heading, speed at the last real frame
        self.ego_blue = item["images"][n - 1, 2].clone()           # the ego is drawn identically in every frame
        self.t = 0                                                 # imagined steps taken
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.z_episode = None                                      # z_seq models: one z for the whole episode
        self.history = [item["images"][i] for i in range(n)]      # real warm-up frames, then imagined ones
        self.plan = []                                             # the actions you took
        # The human's route from the last real frame on, as curvature against distance travelled, so any speed
        # profile follows the same path (curvature indexed by time would bend a faster ego off the route).
        raw = item["actions"].numpy() * self.a_std + self.a_mean
        v = st[n - 1:, 3]
        seg = 0.5 * (v[:-1] + v[1:]) * self.cfg.data.dt                 # metres covered in each logged step
        self.route_s = np.concatenate([[0.0], np.cumsum(seg)])           # arc length at the start of each step
        self.route_k = raw[n - 1:, 1]                                    # curvature of each logged step
        self.s = 0.0                                                     # distance the ego has travelled
        return item["images"][n - 1]

    def reseed(self, seed: int):
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.z_episode = None                                      # z_seq models: one z for the whole episode

    @property
    def beyond_horizon(self) -> bool:
        return self.t >= self.cfg.data.n_pred

    def logged_action(self):
        """The human driver's action at the current step (None past the logged window)."""
        n, i = self.cfg.data.n_cond, self.cfg.data.n_cond - 1 + self.t
        if i >= self.item["actions"].shape[0]:
            return None
        return self.item["actions"][i].numpy() * self.a_std + self.a_mean     # raw (accel, curvature)

    @torch.no_grad()
    def step(self, accel: float | None = None, curvature: float | None = None):
        """Advance 0.2 s. accel in m/s^2, curvature in 1/m; None = use the logged human value (0 past the window).
        Returns the imagined frame (3, 64, 64): red = real road at the new pose, green = predicted others, blue = ego."""
        d, rc = self.cfg.data, self.cfg.render
        logged = self.logged_action()
        acc = accel if accel is not None else (float(logged[0]) if logged is not None else 0.0)
        if curvature is None and accel is None and logged is not None:
            cur = float(logged[1])                  # driving exactly like the human: the logged step, by time
        elif curvature is None:                     # own speed: follow the human's route by distance; past its end,
            # keep turning. (Distance is ambiguous while stopped, which is why the exact-replay case above uses time.)
            j = int(np.searchsorted(self.route_s, self.s + 1e-6, side="right")) - 1   # 1 µm: float rounding at step ends
            cur = float(self.route_k[min(max(j, 0), len(self.route_k) - 1)])
        else:
            cur = curvature
        acc = float(np.clip(acc, -d.accel_clip, d.accel_clip))
        cur = float(np.clip(cur, -d.curvature_clip, d.curvature_clip)) if self.state[3] >= d.v_min_curvature else 0.0

        a_n = torch.tensor((np.array([acc, cur]) - self.a_mean) / self.a_std, dtype=torch.float32, device=self.dev)[None]
        v_n = torch.tensor([[(self.state[3] - self.stats["speed_mean"]) / self.stats["speed_std"]]],
                           dtype=torch.float32, device=self.dev)
        m = self.model
        if m.cfg.deterministic:
            z = m.zero_z(1, self.dev)
        elif m.cfg.z_seq:                           # sampled once per episode, then held
            if self.z_episode is None:
                mu = m.zero_z(1, self.dev)
                self.z_episode = torch.randn(mu.shape, generator=self.gen, device=self.dev)
            z = self.z_episode
        else:
            mu, lv = m.prior_params(self.h)
            z = mu + (0.5 * lv).exp() * torch.randn(mu.shape, generator=self.gen, device=self.dev)
        now = self.history[-1][None].to(self.dev)                                   # current frame (real or imagined)
        self.h, g = m.rollout_step(self.h, z, a_n, v_n, frame=now, prev=now[:, 1:2])
        green = g[0, 0].float().cpu()

        v0 = self.state[3]
        self.state = step_np(self.state, np.array([acc, cur]), d.dt)              # kinematic ego
        self.s += 0.5 * (v0 + self.state[3]) * d.dt
        road = torch.from_numpy(self._road(self.rec).crop(*self.state[:3], rc)).float() / 255.0
        frame = torch.stack([road, green, self.ego_blue])
        self.t += 1
        self.history.append(frame)
        self.plan.append((acc, cur))
        return frame

    def truth(self):
        """The real frame at the current time (only meaningful if you drove like the human), or None past the window."""
        i = self.cfg.data.n_cond - 1 + self.t
        return self.item["images"][i] if i < self.item["images"].shape[0] else None

    def rollout(self, accel, steps: int):
        """From the current state, `steps` ticks of constant acceleration `accel` (None = the human's) along the human's
        route. Returns (green (steps, 64, 64), ego poses (steps, 3) = x, y, heading, road (steps, 64, 64)), and leaves
        the sim at the end."""
        greens, poses, roads = [], [], []
        for _ in range(steps):
            f = self.step(accel, None)
            greens.append(f[1]); poses.append(self.state[:3].copy()); roads.append(f[0])
        return torch.stack(greens), torch.tensor(np.array(poses)), torch.stack(roads)

    # ---- branching --------------------------------------------------------------------------------------------------
    def save(self, name: str):
        self._saved[name] = {"h": self.h.clone(), "state": self.state.copy(), "t": self.t, "s": self.s,
                             "gen": self.gen.get_state(), "z": self.z_episode, "history": list(self.history), "plan": list(self.plan)}

    def restore(self, name: str):
        s = self._saved[name]
        self.h, self.state, self.t = s["h"].clone(), s["state"].copy(), s["t"]
        self.gen.set_state(s["gen"]); self.z_episode = s.get("z"); self.history = list(s["history"]); self.plan = list(s["plan"])
        self.s = s["s"]


# ---------------------------------------------------------------------------------------------------------------------
def to_rgb(frame, scale: int = 6, road_dim: float = 0.6) -> np.ndarray:
    """(3, 64, 64) frame -> (64*scale, 64*scale, 3) uint8 for display (road dimmed so traffic stands out)."""
    f = frame.clone().float()
    f[0] = f[0] * road_dim
    img = (f.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))


def play(ckpt: str, cache_dir: str, data_dir: str, split: str = "val", window: int = 0):
    """Interactive driver for Jupyter (needs `pip install ipywidgets`). Left: imagined. Right: the real log
    (only meaningful while you drive like the human). Keys are buttons: accelerate / coast / brake, steer, step,
    play 1 s, save & restore a branch point, reset, new scene."""
    import ipywidgets as W
    from IPython.display import display

    sim = ImaginedDrive.load(ckpt, cache_dir, data_dir, split)
    state = {"accel": 0.0, "curv": None}
    first = sim.reset(window)
    img_sim, img_truth = W.Image(format="png"), W.Image(format="png")
    info = W.HTML()

    def png(arr):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="png"); return buf.getvalue()

    def show(frame):
        img_sim.value = png(to_rgb(frame))
        tr = sim.truth()
        img_truth.value = png(to_rgb(tr) if tr is not None else np.zeros((384, 384, 3), np.uint8))
        x, y, th, v = sim.state
        warn = " <b style='color:#e67'>beyond the 4 s training horizon</b>" if sim.beyond_horizon else ""
        steer = "follow log" if state["curv"] is None else f"{state['curv']:+.3f} 1/m"
        acc = "follow log" if state["accel"] is None else f"{state['accel']:+.1f} m/s²"
        info.value = (f"scene {sim.window} (rec {sim.rec}) | t = +{sim.t * 0.2:.1f} s | speed {v:4.1f} m/s | "
                      f"accel {acc} | steer {steer}{warn}")

    def set_acc(a):
        def f(_): state["accel"] = a; show(sim.history[-1])
        return f

    def set_curv(c):
        def f(_): state["curv"] = c; show(sim.history[-1])
        return f

    def do_step(n):
        def f(_):
            for _ in range(n):
                fr = sim.step(state["accel"], state["curv"])
            show(fr)
        return f

    def do_reset(new_scene):
        def f(_):
            w = int(np.random.randint(len(sim.ds))) if new_scene else sim.window
            show(sim.reset(w))
        return f

    B = lambda label, cb, style="": (lambda b: (b.on_click(cb), b)[1])(W.Button(description=label, button_style=style))
    rows = [
        W.HBox([B("accelerate +1.5", set_acc(1.5)), B("coast 0", set_acc(0.0)), B("brake −3", set_acc(-3.0), "warning"),
                B("follow log speed", set_acc(None))]),
        W.HBox([B("steer left", set_curv(0.05)), B("follow log path", set_curv(None)), B("straight", set_curv(0.0)),
                B("steer right", set_curv(-0.05))]),
        W.HBox([B("step 0.2 s", do_step(1), "primary"), B("play 1 s", do_step(5), "primary"),
                B("save point", lambda _: sim.save("A")), B("back to point", lambda _: (sim.restore("A"),
                                                                                         show(sim.history[-1]))),
                B("reset", do_reset(False)), B("new scene", do_reset(True), "info")]),
    ]
    display(W.VBox([info, W.HBox([W.VBox([W.HTML("<b>imagined</b>"), img_sim]),
                                  W.VBox([W.HTML("<b>real log</b> (valid only if you drive like the human)"),
                                          img_truth])])] + rows))
    show(first)
    return sim
