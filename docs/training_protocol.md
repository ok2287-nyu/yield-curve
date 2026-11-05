# Training protocol: deterministic stage, then stochastic stage, then a manual check

The same recipe for every encoder (`cnn`, `vit`, `vit_mae`). Only the encoder differs between the three runs:
same code, same seed, same number of steps, same flags. Every command below runs from the repo on HPC:

```bash
source /scratch/$USER/yield/venv/bin/activate
cd /scratch/$USER/yield/roundabout_wm
git pull origin main
python tests/test_world_model.py | tail -3          # every line "ok"
```

Lines with `sbatch`, `squeue`, `git`, `pip` or `python - <<'EOF'` go in the **terminal**, not in a Jupyter cell.

---

## 0. Before the first run (once per person)

1. **Your own cache matches the reference.** The end of your `build_cache` job output must show exactly the numbers
   in the README's "Teammate setup" table (22 recordings, 10,428 ego tracks, 793,957 frames, the same `stats.json`).
   If anything differs, stop and tell the team: the encoder comparison is only fair on identical data.
2. **`vit_mae` only:** MAE pretraining first (`sbatch scripts/hpc/pretrain_mae.sbatch`), which writes
   `runs/mae/encoder.pt`.
3. **W&B (optional):** `pip install wandb && wandb login`, and `export WANDB_ENTITY=yield-curve` in `~/.bashrc`.
   Drop `WANDB=1` from the commands below if you do not use it.

Run names below use your encoder as a prefix: `cnn_det`, `vit_det`, `vit_mae_det`, then `cnn_sto`, `vit_sto`,
`vit_mae_sto`. Use a **new run name for every new experiment**: re-using a name resumes that run from its
`last.pt`.

---

## 1. Stage 1: the deterministic model (50,000 steps)

The model without the latent z: it learns the conditional mean (blurry where the future is uncertain, but it keeps
every visible car). It is the base that stage 2 starts from, and the baseline every later model is compared with.

```bash
# cnn
sbatch --export=ALL,ENCODER=cnn,RUN=cnn_det,STEPS=50000,WANDB=1,EXTRA_ARGS="--deterministic" scripts/hpc/train_wm.sbatch
# vit
sbatch --export=ALL,ENCODER=vit,RUN=vit_det,STEPS=50000,WANDB=1,EXTRA_ARGS="--deterministic" scripts/hpc/train_wm.sbatch
# vit_mae (encoder initialised from MAE; its learning rate is automatically 0.1x)
sbatch --export=ALL,ENCODER=vit,RUN=vit_mae_det,STEPS=50000,WANDB=1,INIT=runs/mae/encoder.pt,EXTRA_ARGS="--deterministic" scripts/hpc/train_wm.sbatch
```

Time: the CNN took 2 h 05 min on one A100 (0.147 s/step). Check your own `s/step` in the first minutes of
`wm_<jobid>.out`.

**Check when it finishes** (terminal; replace `cnn_det` with your run):

```bash
python - <<'EOF'
import pandas as pd
d = pd.read_csv("runs/cnn_det/log.csv"); v = d[d.split == "val"]
cols = ["step"] + [f"{m}@{k}" for k in (5, 10, 20) for m in ("prior", "empty", "warp", "mass")] + ["act_sens", "mse_warm"]
pd.set_option("display.width", 250); print(v[cols].iloc[::5].round(4).to_string(index=False))
EOF
```

Stage 1 is good when, at the last validation:

| Check | Pass | `cnn_det` (reference) |
|---|---|---|
| `prior@5`, `prior@10`, `prior@20` | below `empty@` and `warp@` at the same step | 0.0051 / 0.0087 / 0.0108 vs empty 0.0167 / 0.0164 / 0.0148 |
| `mass@5`, `mass@10`, `mass@20` | close to 1 (no fading) | 0.99 / 0.98 / 0.96 |
| `act_sens` | clearly above 0 | 0.023 |
| curve | flat over the last ~10k steps (if still falling, train longer) | flat from ~31k |

Then the baselines on the same windows (CPU, ~15 min; the model row must beat constant velocity and constant turn
rate at 2 s and 4 s):

```bash
python scripts/eval_baselines.py --cache ../round_cache --data_dir ../round_data --ckpt runs/cnn_det/best.pt
```

The table now has a `ctrv_blur` row: constant turn rate, blurred and scaled per horizon step, both tuned on 1,000
**train** windows to minimise MSE (the tuned sigma/alpha are printed and saved in the csv). MSE rewards hedging, so
this is the fair kinematic baseline for a model that predicts a blurry mean. Report the model against it.

---

## 2. Stage 2: the stochastic model (50,000 steps, warm-started from stage 1)

Adds the latent z (learned prior + posterior) on top of the stage-1 model, so samples can show *different* futures.
The recipe below came out of the Nov 4 ablation (it is the only one that used z without collapsing):

- `--init_from <stage 1>/best.pt` start from the deterministic model
- `--pz_init_scale 0.01` z's input layer starts small but not zero (zero blocks all learning of z)
- `--kl_warmup_steps 2000` no KL penalty for 2k steps, then a ramp to beta (prevents posterior collapse)
- `--beta 1e-4`

```bash
STAGE2="--pz_init_scale 0.01 --kl_warmup_steps 2000 --beta 1e-4"
# cnn
sbatch --export=ALL,ENCODER=cnn,RUN=cnn_sto,STEPS=50000,WANDB=1,EXTRA_ARGS="--init_from runs/cnn_det/best.pt $STAGE2" scripts/hpc/train_wm.sbatch
# vit
sbatch --export=ALL,ENCODER=vit,RUN=vit_sto,STEPS=50000,WANDB=1,EXTRA_ARGS="--init_from runs/vit_det/best.pt $STAGE2" scripts/hpc/train_wm.sbatch
# vit_mae (keep the encoder's learning rate at 0.1x, as in stage 1)
sbatch --export=ALL,ENCODER=vit,RUN=vit_mae_sto,STEPS=50000,WANDB=1,EXTRA_ARGS="--init_from runs/vit_mae_det/best.pt --enc_lr_mult 0.1 $STAGE2" scripts/hpc/train_wm.sbatch
```

Train long: 50k steps, the same as stage 1 (about 2.8 h for the CNN). PPUU trained its stochastic model for at least
as long as its deterministic one; 10k-step stage-2 runs were not long enough to judge.

**Which checkpoint.** Since the `eval-tools` merge, a stochastic run saves as `best.pt` the checkpoint with the
lowest **validation ELBO** (posterior MSE + warm-up MSE + beta x KL, with the run's final beta; logged as `elbo`), not
the lowest prior MSE. Prior MSE is the wrong criterion for a stochastic model: a single sample is *supposed* to have a
higher MSE than the mean, so it picked an early, nearly deterministic checkpoint (`ws_both/best.pt` was step 1,000
of 10,000). Deterministic runs still use prior MSE (`--best_metric auto`; force either with `--best_metric prior|elbo`).
**Runs trained before this change** (`ws_*`, `ws_both_long`): their `best.pt` is still the old choice, so use
`last.pt` for them.

**Numbers to check** (terminal):

```bash
python - <<'EOF'
import pandas as pd
d = pd.read_csv("runs/cnn_sto/log.csv"); v = d[d.split == "val"]
cols = ["step"] + [f"{m}@{k}" for k in (5, 20) for m in ("prior", "post", "mass", "spread")] + ["kl", "act_sens"]
pd.set_option("display.width", 250); print(v[cols].iloc[::5].round(4).to_string(index=False))
EOF
```

| Check | Pass | Fail looks like |
|---|---|---|
| `kl` | clearly above 0 (several nats) and not falling towards 0 | ~0.01: z collapsed, the model is deterministic again |
| `spread@20` | clearly above 0 | ~0.000x: all samples identical |
| `post@k` < `prior@k` | yes (the posterior saw the future) | equal: z is unused |
| `mass@20` | rising towards 1 over training | stuck well below 1: samples fade |
| `act_sens` | clearly above 0 | ~0: the model ignores the ego |
| `elbo` | falling, then flat | rising while `kl` rises: beta too small, z is memorising |

A single prior sample is **expected** to have a higher MSE than the stage-1 model. That is not a failure: each sample
commits to one future, the mean hedges. Do not judge stage 2 by beating stage 1 on MSE.

---

## 3. Manual check: look at the predictions

The numbers above cannot tell "diverse, plausible futures" from "noise". Look at the pictures.

```bash
python scripts/compare_models.py --cache ../round_cache --det runs/cnn_det/best.pt \
    --stoch runs/cnn_sto/best.pt --scenes 0 72 145 218 290 363 --k 4 --out_dir runs/compare_cnn_sto
```

(Several stochastic runs can be compared at once: list more checkpoints after `--stoch`; `last.pt` for runs from
before the ELBO change.) It writes one sheet per
scene, `runs/compare_cnn_sto/compare_<scene>.png`, downloadable from the portal's Files view, with one column per
0.4 s and these rows:

```
truth                the real future
deterministic        the stage-1 model (the mean)
<run> post           z from the posterior, which has seen the next frame
<run> prior #1..#4   four open-loop samples (what a planner would see)
<run> spread         where the four samples disagree (white = disagree)
```

Go through every sheet and answer each question **by looking**:

1. **Is the posterior the closest row to the truth?** Its cars should sit where the truth's cars are, at every
   column, including cars the deterministic row only smears. The script also prints `post` (its MSE) per scene: it
   should be the smallest number on the line. If the posterior is no closer to the truth than the deterministic row,
   z carries nothing.
2. **Do the prior samples differ substantially from each other?** Compare prior #1 to #4 column by column: a car
   further along in one sample than another, a car that merges in one sample and waits in another, a car that appears
   in only some samples. If the four rows look the same, z is unused under the prior (check `kl` and `spread`).
3. **Do they differ in the right places?** The spread row should light up on specific, contested road users (a car
   at an entry, a car deciding to exit) and stay dark on predictable ones (the car directly behind you, a car driving
   steadily along the ring). Bright everywhere = noise, not uncertainty.
4. **Are the samples plausible?** Each sample's cars stay car-shaped and keep their brightness over the 4 s. Cars
   that dissolve into nothing along a prior row = fading (also visible as low `mass`). A car that is clearly present in
   the posterior row but missing from every prior sample = the prior cannot produce that future.
5. **Is one of the samples close to what really happened?** For the scenes where the truth does something specific
   (merges, waits), at least one prior sample should show roughly that. The printed `best` (best of 4) should be
   clearly lower than `mean` (average of the 4) in scenes where samples differ.

### Does the model predict that others react to the ego? (reaction map)

```bash
python scripts/reaction_map.py --ckpt runs/cnn_sto/best.pt --cache ../round_cache --data_dir ../round_data \
    --scene 145 --a brake --b go --seconds 4 --k 8
```

Same start, two plans (`brake` -3 m/s², `go` +1.5 m/s², `coast`, `log` = the human, or a number). Plan B's prediction
is warped into plan A's ego frame at every step (the camera moved differently), so what remains in the `B - A` row is
the predicted change in the *other* road users: green = more traffic under B, red = less. For a stochastic model both
plans are the mean of `--k` prior samples with the same seeds. It prints the reaction score (mean |B - A| over the
area both crops share) every 0.4 s. Pick scenes with a car near an entry: a reactive model should show that car
yielding (red ahead of it) or going (green) depending on the ego's plan. Caveat: the model learned from logs where the
human ego reacted to others, so a difference can be correlation; look at whether it appears only after the plans
diverge, and only on cars that interact with the ego.

**Many scenes at once** (no pictures; ~5 min on CPU for the deterministic model, ~8x that for a stochastic one):

```bash
python scripts/reaction_map.py --ckpt runs/cnn_det/best.pt --cache ../round_cache --data_dir ../round_data \
    --batch 200 --pairs brake:go,brake:coast,coast:go
```

Per pair and 0.4 s: `reaction` (how much the predicted traffic changes with the plan, in cars), `net` (traffic
created + or deleted - by the plan, in cars) and `|net|/reaction` (0 = cars only move: following, yielding;
1 = cars only appear or vanish). A plan can move other cars but cannot create or delete them, so `|net|` is the
leakage: the plan used as evidence about the traffic. It ends with the five scenes with the most leakage at 2 s:
look at those with the single-scene command.

**Leakage, measured where no reaction is possible.** A negative `net` above can also be a real reaction (a car up
the ring yields and enters the view later). `scripts/leakage.py` only scores road users the ego cannot affect within
4 s: seen during the warm-up, moving away from the ego under both plans, and whose path never comes within 8 m of the
ego's (~10 min on CPU for 300 scenes):

```bash
python scripts/leakage.py --ckpt runs/cnn_det/best.pt --cache ../round_cache --data_dir ../round_data --n_scenes 300
```

`receding` (|B - A| around those road users, per unit of their true area) should sit at `floor` (the same measure for a
model with no leakage). `interacting` is the same measure around the other seen road users, for scale. Check
`coverage` (~1 means the model's cars are where the region expects them) and `receding area` (steps with too little
receding traffic print `-`).

Known limitation, not a model failure: road users that enter the 48 m crop after the 2 s warm-up (the model never saw
them) are missing from every row except the posterior and the truth. At 4 s this is ~43% of the true car area (see
`scripts/fov_popin.py`). Scene 363 is an example.

**Write down for each scene:** posterior closest to truth (yes / no), samples differ (yes / no, where), plausible
(yes / fading / noise), one sample close to the truth (yes / no). Share that list and two or three sheets with the
team, **inside the team only**: the images are rendered from rounD data.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Job stays in `CF` for ~13-15 min, then `CANCELLED`, no `.out` file | GCP has no A100 free for the cloud node (HPC support, Nov 5: A100s are scarce at the moment) | resubmit and be patient. **Do not use the L4 partition `g2-standard-12`**: HPC support says L4s are very hard to get there |
| `ModuleNotFoundError: numpy` | the venv is not active (new portal session) | `source /scratch/$USER/yield/venv/bin/activate` |
| `ls: cannot access 'wm_*.out'` | you are in `~` | `cd /scratch/$USER/yield/roundabout_wm` |
| W&B runs stopped appearing | `wandb offline` was run in the repo folder (e.g. by pasting log text into the shell) | `wandb online` in the repo folder |
| Stage 2 `kl` ~0.01 from the start | posterior collapse | check `--pz_init_scale 0.01 --kl_warmup_steps 2000` are in `EXTRA_ARGS` |
| A rerun starts at a high step number | the run name already exists, so it resumed | use a new `RUN=` name |

**Cloud Bursting rules (HPC support, Nov 5).** A GCP node **shuts down automatically if its resources are
underused for 20 minutes**, so never hold a GPU job open for interactive or CPU-only work (evaluation scripts,
`reaction_map.py`, `leakage.py`, `compare_models.py` run fine on the CPU Jupyter session). An A100 costs about $2 per
GPU-hour. Because A100s are scarce, submit training runs as soon as they are ready rather than in a batch later.

**GPU budget** (300 GPU-hours each; one encoder's two stages cost about 5-6 A100-hours). Your usage so far:

```bash
sacct -u $USER -S 2026-10-01 -X --format=ElapsedRaw,AllocTRES%60 -n -P | python3 -c "
import sys, re
h = sum(int(l.split('|')[0]) * int(m.group(1)) / 3600 for l in sys.stdin
        if (m := re.search(r'gres/gpu=(\d+)', l)) and l.split('|')[0])
print(f'GPU-hours used: {h:.1f} of 300')"
```
