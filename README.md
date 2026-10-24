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

## Teammate setup (NYU HPC Cloud Bursting)

Rules first:
- **Never share rounD data**, not even inside the team (licence). Everyone downloads with their own approved
  access and builds their own cache. `.gitignore` keeps data, caches, renders and checkpoints out of git; don't
  force-add them.
- Cloud Bursting only (account `csci_ga_2271-2026fa`); nothing on the Torch cluster.
- Shared code is identical across runs. The only file you change for your run is `roundabout_wm/models/encoders.py`.

Paths below use `$USER`, so they work for any netid; the sbatch scripts expect exactly this layout:

```
/scratch/$USER/yield/
  round_data/      NN_tracks.csv, NN_tracksMeta.csv, NN_recordingMeta.csv, NN_background.png
  venv/            Python environment
  roundabout_wm/   this repo
  round_cache/     built by step 4
```

**1. Data.** Upload your own rounD zip (e.g. `round-dataset-v1.1.zip`) through the portal's Files view, then
put the contents of its `data/` folder into `round_data/`:

```bash
mkdir -p /scratch/$USER/yield/raw /scratch/$USER/yield/round_data
cd /scratch/$USER/yield
unzip ~/round-dataset-v1.1.zip -d raw            # adjust to where you uploaded it
cp raw/*/data/* round_data/                      # adjust if the zip's folder layout differs
ls round_data | head                              # expect 00_background.png, 00_recordingMeta.csv, ...
```

**2. Environment.** (Ojaswi's: Python 3.9, torch 2.8.0+cu128.)

```bash
cd /scratch/$USER/yield
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas opencv-python-headless matplotlib imageio pillow
```

**3. Code.**

```bash
cd /scratch/$USER/yield
git clone https://github.com/<owner>/<repo>.git roundabout_wm
cd roundabout_wm
python tests/test_conventions.py && python tests/test_world_model.py      # all "ok"
```

If the clone fails (no GitHub access from the node), download the repo as a zip on your laptop and upload it.

**4. Build your cache** (CPU partition, no GPU hours; about an hour; resubmit if cut short, finished
recordings are skipped):

```bash
sbatch scripts/hpc/build_cache.sbatch      # output in round_cache_<jobid>.out
```

**5. Check it matches.** The end of the job output (from `scripts/hpc/check_cache.py`) must show exactly the
reference numbers below. If anything differs, stop and tell the team before training: the encoder comparison is
only fair if all three caches are identical.

| Quantity | Reference (Ojaswi, job 9435) |
|---|---|
| Recordings | 22: 2 ... 23 |
| Split | train 2-17, val 18-20, test 21-23 |
| Ego tracks / frames | 10,428 / 793,957 (about 9.8 GB on disk) |
| Windows at n_pred = 20 | train 70,873, val 17,374, test 14,309 |
| `stats.json` | speed 6.29 +/- 3.16 m/s; acceleration 0.31 +/- 0.91 m/s^2; curvature 0.0026 +/- 0.032 1/m |

**6. Add your encoder** in `roundabout_wm/models/encoders.py` and register it in `build_encoder`. It must pass
`check_encoder_contract` in `tests/test_world_model.py`: input one frame `(B, 3, 64, 64)` in [0, 1], output
`(B, 256, 8, 8)` where cell `(r, c)` is the 8x8-pixel patch at row r, column c; deterministic in train mode
(no dropout in the encoder, decision 17). Add a test that calls `check_encoder_contract` on it.

**7. Train** only after the `cnn` run has passed the gate (`docs/experiments.md`):

```bash
sbatch --export=ALL,ENCODER=vit,RUN=vit scripts/hpc/train_wm.sbatch
sbatch --export=ALL,ENCODER=vit,RUN=vit_mae,INIT=runs/mae/encoder.pt scripts/hpc/train_wm.sbatch
```

Use the same `STEPS` as the `cnn` run. Check GPU use on a running job with
`srun --jobid=<id> --overlap nvidia-smi` (aim for > 50%; GPU nodes idle for 20 minutes are shut down).

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
