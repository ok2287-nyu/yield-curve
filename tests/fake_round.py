"""A tiny synthetic recording in rounD's file format, built into a cache with the real build code (tests only).

Straight road along +x at y = -50 m, everyone at 6 m/s, 60 steps of 0.2 s:
    1  ego       starts at x = 10
    2  leader    20 m ahead of the ego (centre to centre)
    3  follower  12 m behind
    4  parked    a car 40 m to the side (never in the way)
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from roundabout_wm.build import build_recording
from roundabout_wm.config import Config
from roundabout_wm.round_io import load_recording

V, N = 6.0, 60


def make(root, rec_id=0):
    cfg = Config()
    raw, cache = os.path.join(root, "raw"), os.path.join(root, "cache")
    os.makedirs(raw, exist_ok=True)
    ortho = 0.01                                                      # background: 0.1 m per px after scale-down 10
    pd.DataFrame([dict(recordingId=rec_id, locationId=0, frameRate=25, orthoPxToMeter=ortho)]).to_csv(
        os.path.join(raw, f"{rec_id:02d}_recordingMeta.csv"), index=False)
    pd.DataFrame([dict(trackId=k, **{"class": "car"}, width=1.8, length=4.5) for k in (1, 2, 3, 4)]).to_csv(
        os.path.join(raw, f"{rec_id:02d}_tracksMeta.csv"), index=False)
    img = np.full((1500, 1500), 60, np.uint8)
    img[480:520, :] = 200                                             # the road (rows of y = -48 .. -52 m)
    cv2.imwrite(os.path.join(raw, f"{rec_id:02d}_background.png"), img)
    rows = []
    for j in range(N):
        f, t = j * cfg.data.step, j * cfg.data.dt
        for k, x0, y in ((1, 10.0, -50.0), (2, 30.0, -50.0), (3, -2.0, -50.0), (4, 20.0, -10.0)):
            v = 0.0 if k == 4 else V
            rows.append(dict(trackId=k, frame=f, xCenter=x0 + v * t, yCenter=y, heading=0.0, lonVelocity=v,
                             lonAcceleration=0.0))
    pd.DataFrame(rows).to_csv(os.path.join(raw, f"{rec_id:02d}_tracks.csv"), index=False)
    build_recording(load_recording(raw, rec_id, step=cfg.data.step), cfg, os.path.join(cache, f"rec_{rec_id:02d}"),
                    verbose=False)
    with open(os.path.join(cache, "split.json"), "w") as fh:
        json.dump({"train": [rec_id], "val": [rec_id], "test": [rec_id]}, fh)
    return cfg, raw, cache


def ego_window(env, track=1, start_row=0):
    """Index of the window of `track` starting `start_row` rows into it."""
    for i in range(len(env.ds)):
        rec, tid, off = env._locate(i)
        if tid == track and off == start_row:
            return i
    raise KeyError((track, start_row))
