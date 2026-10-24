# Roundabout world model

Turns rounD drone recordings into ego-centric bird's-eye-view training windows, and trains an
action-conditioned stochastic world model on them (spatial RSSM-style, learned prior, kinematic ego).

## Layout

```
roundabout_wm/
  config.py       every design decision (rates, window, crop, clipping) in one place
  round_io.py     reading rounD files; coordinate conventions
  kinematics.py   actions (acceleration, curvature) <-> kinematic ego update (numpy + torch)
  render.py       ego-centric 3x64x64 renderer: red = road, green = other road users, blue = ego
  build.py        render + cache one recording; split by recording
  dataset.py      PyTorch Dataset over the cache; normalisation stats (rollout length chosen at load time)
  models/
    encoders.py   per-frame encoders behind --encoder (cnn here; vit added by its owner); contract 3x64x64 -> 256x8x8
    world_model.py  v1: encoder -> ConvGRU -> green-only decoder; learned prior, z-dropout, stop-gradient on prior samples
  wm_eval.py      warp-previous baseline, counterfactual action swaps
scripts/
  check_overlay.py   step 1: trajectories on the aerial image
  check_actions.py   step 2: integrate extracted actions, compare with recorded paths
  render_track.py    step 4: one ego track as a GIF next to the aerial view
  build_cache.py     step 5: cache all Neuweiler recordings, write split.json and stats.json
  train_wm.py        train a world model (checkpoints + auto-resume, validation with baselines)
  hpc/               Cloud Bursting job scripts (cache build on CPU, training on A100)
tests/
  test_conventions.py  sign conventions (ahead = up, left = left), kinematics round-trip, torch == numpy
  test_world_model.py  encoder contract, shapes, action-only-into-Pred, no future leakage, stop-gradient, warp baseline vs renderer
docs/decisions.md      what was decided and why
docs/experiments.md    the CNN / ViT / ViT+MAE comparison: who runs what, what to compare
```

## Run

```bash
pip install numpy pandas opencv-python-headless matplotlib imageio torch
python tests/test_conventions.py
python scripts/check_overlay.py --data_dir $DATA --rec 17
python scripts/check_actions.py --data_dir $DATA --rec 17 --track 30
python scripts/render_track.py  --data_dir $DATA --rec 17 --track 30
python scripts/build_cache.py   --data_dir $DATA --out_dir $SCRATCH/round_cache
```

`$DATA` is the folder holding `NN_tracks.csv`, `NN_tracksMeta.csv`, `NN_recordingMeta.csv`, `NN_background.png`.
Each person uses their own copy (rounD licence: no sharing inside the group).

## Using the dataset

```python
from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows, load_stats
import json
cfg = Config()
split = json.load(open(f"{cache}/split.json"))
stats = load_stats(f"{cache}/stats.json")
train = RounDWindows(cache, split["train"], cfg, stats)
item = train[0]
# images (30, 3, 64, 64) in [0,1]; speed (30, 1); actions (29, 2); ego_state (30, 4)
# warm up on images[:10]; predict the green channel of images[10:] given actions[9:] and speed[9:]
```

## Numbers from recording 17 (Neuweiler)

| Check | Result |
|---|---|
| Background scale | 0.1016 m/px (orthoPxToMeter x 10); 0.03% of track points fall off the image |
| Action round trip, 3 s open loop | position error median 0.41 m, 95th pct 1.01 m, max 2.20 m (597 cars) |
| Cache | 585 ego tracks, 48,267 frames, 7,088 windows, 567 MB, 18 s to build |
| Normalisation (rec 17 only) | speed 5.8 +/- 3.4 m/s; accel 0.38 +/- 0.94 m/s^2; curvature 0.002 +/- 0.033 1/m |
