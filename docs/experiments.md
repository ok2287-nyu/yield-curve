# World-model experiments: CNN vs ViT vs ViT + MAE

**Question:** which per-frame encoder gives the best world model *for planning* on rendered roundabout traffic?
Everything except the encoder is identical: ConvGRU dynamics, learned prior with stop-gradient on prior samples,
z-dropout, green-only decoder, MSE + 1e-6 KL, 10 warm-up + 20 predicted steps, same data, seed and number of steps.

| Run | Encoder | Init | Command (in `scripts/hpc/`) |
|---|---|---|---|
| `cnn` | strided CNN 64-128-256 | scratch | `sbatch --export=ALL,ENCODER=cnn,RUN=cnn train_wm.sbatch` |
| `vit` | ViT (owner adds it to `models/encoders.py`) | scratch | `sbatch --export=ALL,ENCODER=vit,RUN=vit train_wm.sbatch` |
| `vit_mae` | same ViT | MAE on single train frames (owner's script); encoder lr x0.1 | owner's MAE job, then `sbatch --export=ALL,ENCODER=vit,RUN=vit_mae,INIT=runs/mae/encoder.pt train_wm.sbatch` |
| `cnn_nosg` (ablation) | CNN | scratch, `--no-prior_sg` | `sbatch --export=ALL,ENCODER=cnn,RUN=cnn_nosg,EXTRA_ARGS=--no-prior_sg train_wm.sbatch` |

Adding an encoder: implement it in `models/encoders.py`, register it in `build_encoder`, and make it pass
`check_encoder_contract` in `tests/test_world_model.py`. Do not change any other shared file for one run.

Parameter counts are written to `runs/<run>/params.json`. If `vit` wins, also run a ViT with about the CNN's
parameter count before claiming architecture is the reason.

## Two-stage training (current recipe for the CNN run)

```bash
# stage 1: deterministic world model (the conditional mean)
sbatch --export=ALL,ENCODER=cnn,RUN=cnn_det,STEPS=10000,WANDB=1,EXTRA_ARGS="--deterministic" scripts/hpc/train_wm.sbatch
# stage 2: stochastic, warm-started from stage 1
sbatch --export=ALL,ENCODER=cnn,RUN=cnn_ws,STEPS=10000,WANDB=1,EXTRA_ARGS="--init_from runs/cnn_det/best.pt --beta 1e-4" scripts/hpc/train_wm.sbatch
```

Stage 1 should already beat empty and warp and keep green mass near 1 (blurry but complete). If it fades too,
the problem is in Pred/Dec, not in z. Stage 2 should improve on stage 1 without a large posterior/prior gap.

## Order and gate

1. `cnn` first, alone. Gate before anyone trains `vit` / `vit_mae`: beats predict-empty and warp-previous at
   1, 2 and 4 s (`prior@5/10/20` vs `empty@` and `warp@`), KL not flat at 0, act_sens clearly above 0.
   Also watch the posterior/prior gap, prior sigma and green mass (below).
2. Then `vit` and MAE pretraining in parallel (different people's GPU hours), then `vit_mae`.

## What to compare (`runs/<run>/log.csv`, the printed VAL blocks, `samples_*.png`)

| Metric | Meaning | Good |
|---|---|---|
| prior@k | open-loop MSE at step k, z from the learned prior, dropout off | below empty and warp; grows slowly |
| post@k, gap | MSE with posterior z; gap = prior - post | a large, persistent gap = posterior shortcut |
| empty@k / copy@k / warp@k | predict nothing / repeat last frame / warp last frame by logged ego motion | model beats all three |
| mass@k | predicted / true green mass | stays near 1; falling = cars fading |
| kl@k | KL(q \|\| p) per step | not flat at 0 |
| std@k | mean prior sigma | not collapsing to ~0; larger in uncertain scenes |
| act_sens | mean change in prediction when future actions are swapped (speeds re-integrated) | clearly above 0 |
| samples_*.png | rows: truth, 3 prior samples, prior sigma map (white = sigma >= 1.5) | cars stay sharp; samples differ where the future is uncertain |

## Field of view: vehicles the model could not have seen

```bash
python scripts/fov_popin.py --data_dir ../round_data            # CPU only, a few minutes; writes fov_popin.csv
```

Share of windows where a road user that was never visible during warm-up enters the crop by 1 / 2 / 4 s, its
share of the visible box area, and how often such a pop-in comes within 15 m of the ego, for the current crop,
the same crop with the ego lower (row 55), and a 96 m crop. A bigger crop has longer edges, so more far-away
pop-ins: compare the area share and the close-pop-in rate, not just the window count.

## Looking at the predictions

```bash
python scripts/inspect_wm.py --ckpt runs/<run>/best.pt --cache $C          # 6 val scenes with traffic
```

Writes `runs/<run>/inspect_<step>/scene_<idx>.png` (+ `.gif`, `summary.csv`): truth, posterior, K prior samples,
their mean and spread, and brake / speed-up counterfactuals. The script's docstring maps each visual pattern to a
cause (z too weak, z ignored under the prior, prior cannot produce the right z, drift over the horizon, actions
ignored). The images are rendered rounD data: keep them local.

## Budget

300 GPU hours per person on 1xA100 (`c12m85-a100-1`). Read `s_per_step` and `gpu_util` from the first minutes of
the `cnn` run, then set `--steps` (same for all runs). GPU utilisation under ~50% means the loader is the
bottleneck (more workers, or copy the cache to node-local disk).
