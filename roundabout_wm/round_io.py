"""Reading rounD recordings.

File layout (one set per recording id NN, e.g. 17):
    NN_recordingMeta.csv  one row: frameRate, locationId, orthoPxToMeter, ...
    NN_tracksMeta.csv     one row per road user: trackId, class, width, length, ...
    NN_tracks.csv         one row per road user per frame: xCenter, yCenter, heading, ...
    NN_background.png     aerial photo without road users

Conventions (verified on recording 17):
    - x, y in metres; origin at the top-left of the image; y points UP, so every y is negative.
    - heading in degrees, counter-clockwise from +x (matches atan2(yVelocity, xVelocity)).
    - lonVelocity is the speed along the vehicle's nose (can be slightly negative from noise).
    - the lateral axis (latVelocity, latAcceleration) points to the vehicle's RIGHT.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRACK_COLS = ["trackId", "frame", "xCenter", "yCenter", "heading", "lonVelocity", "lonAcceleration"]


@dataclass
class Recording:
    rec_id: int
    meta: dict
    tracks_meta: pd.DataFrame       # indexed by trackId
    tracks: pd.DataFrame            # downsampled; columns: trackId, frame, x, y, heading (rad), speed
    background_path: str

    @property
    def location_id(self) -> int:
        return int(self.meta["locationId"])

    def background_m_per_px(self, scale_down: int) -> float:
        return float(self.meta["orthoPxToMeter"]) * scale_down


def _path(data_dir: str, rec_id: int, suffix: str) -> str:
    return os.path.join(data_dir, f"{rec_id:02d}_{suffix}")


def list_recordings(data_dir: str, location_id: int | None = None) -> list[int]:
    """Recording ids found in data_dir, optionally only those at one location."""
    ids = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*_recordingMeta.csv"))):
        meta = pd.read_csv(p).iloc[0]
        if location_id is None or int(meta["locationId"]) == location_id:
            ids.append(int(meta["recordingId"]))
    return ids


def load_recording(data_dir: str, rec_id: int, step: int = 5) -> Recording:
    """Load one recording and keep every `step`-th frame.

    Frames are kept on a global grid (frame % step == 0) so every vehicle shares the same timestamps.
    """
    meta = pd.read_csv(_path(data_dir, rec_id, "recordingMeta.csv")).iloc[0].to_dict()
    tracks_meta = pd.read_csv(_path(data_dir, rec_id, "tracksMeta.csv")).set_index("trackId")

    tr = pd.read_csv(_path(data_dir, rec_id, "tracks.csv"), usecols=TRACK_COLS)
    tr = tr[tr["frame"] % step == 0].copy()
    tr = tr.rename(columns={"xCenter": "x", "yCenter": "y"})
    tr["heading"] = np.deg2rad(tr["heading"].to_numpy())
    tr["speed"] = np.clip(tr["lonVelocity"].to_numpy(), 0.0, None)  # tiny negative values are noise
    tr = tr[["trackId", "frame", "x", "y", "heading", "speed"]]
    tr = tr.sort_values(["trackId", "frame"]).reset_index(drop=True)

    return Recording(rec_id, meta, tracks_meta, tr, _path(data_dir, rec_id, "background.png"))


def world_to_background_px(x, y, m_per_px: float):
    """World metres -> (col, row) in the background PNG. Rows count downward, so y is negated."""
    return np.asarray(x) / m_per_px, -np.asarray(y) / m_per_px
