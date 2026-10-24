"""All design decisions in one place. Change values here, not inside the modules.

Each value has a one-line reason; keep docs/decisions.md in sync with this file.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    # rounD records at 25 Hz. We keep every 5th frame -> 5 Hz, dt = 0.2 s.
    source_hz: int = 25
    step: int = 5

    # Window = n_cond frames of context + n_pred frames to predict (5 Hz).
    n_cond: int = 10          # 2.0 s of past
    n_pred: int = 20          # 4.0 s of future (decided; the single source for train_wm.py --n_pred)
    window_stride: int = 5    # start a new window every 1.0 s of an ego track
    # Cache-time rules, frozen at the values the first cache (job 9435) was built with, so every teammate's cache
    # holds the same ego tracks and the same stats.json. Training windows are rebuilt at load time with n_pred.
    cache_min_track_len: int = 25   # ego tracks shorter than this are not cached

    # Which tracks may be the ego.
    ego_classes: tuple = ("car",)

    # Neuweiler is locationId 0 (most recordings). Split by whole recordings.
    location_id: int = 0
    n_val_recordings: int = 3
    n_test_recordings: int = 3

    # Action extraction / clipping.
    v_min_curvature: float = 0.5   # m/s; below this, curvature is set to 0 (undefined when stopped)
    accel_clip: float = 6.0        # m/s^2; rounD's lonAcceleration lies in about [-5.8, 7.0]
    curvature_clip: float = 0.3    # 1/m; tightest exit turns are well inside this

    # The background PNG is the orthophoto downscaled by this factor
    # (drone-dataset-tools, data/visualizer_params/visualizer_params.json, "round").
    background_scale_down: int = 10

    @property
    def dt(self) -> float:
        return self.step / self.source_hz

    @property
    def window_len(self) -> int:
        return self.n_cond + self.n_pred


@dataclass(frozen=True)
class RenderConfig:
    size: int = 64            # output image is size x size
    m_per_px: float = 0.75    # 64 px * 0.75 = 48 m square
    ego_row: int = 43         # ego centre sits 43 px from the top: ~32 m ahead, ~16 m behind
    supersample: int = 4      # draw cars at 4x and area-average down -> soft edges, sub-pixel accuracy

    @property
    def ego_col(self) -> float:
        return (self.size - 1) / 2.0


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
