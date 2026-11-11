# Stage 2: what we tried for 4 s imagination, and why the horizon is 2 s (Nov 4–11)

All numbers: CNN encoder, 20k-step stage-2 runs warm-started from `cnn_det`, 500 validation windows (closed-window
numbers: the 44 closed windows), one prior sample unless stated. Scripts: `ladder.py`, `closed_mass.py`,
`latent_eval.py`, `eval_sharpness.py`, `compare_models.py`.

## Where seen road users get lost (ladder.py)

| rung | kept | sharp |
|---|---|---|
| reconstruction (decode right after reading the real frame) | 1.00 | 0.98 |
| one step, prior z | 1.00 | 0.88–0.94 |
| 20-step rollout, posterior z | 0.99–1.06 | 0.72–0.89 |
| 20-step rollout, prior z (imagination) | **0.62–0.78 at 4 s** | 0.61–0.84 |

Encoder, grid, decoder, one-step dynamics and memory with the right latent all work. Road users are lost only over
consecutive steps of the model's own latents: kept ≈ 0.98 / 0.90 / 0.70 after 1 / 2 / 4 s. One posterior step
every 6 restores them. The prior *mean* fades as much as prior samples, so sampling noise is not the cause.
Closed windows (no road user enters or leaves): prior-sample mass 0.80 at 4 s, so the loss is real, not only cars
drawn in the wrong place.

## Variants (prior samples at 4 s)

| run | change | kept | lost | sharp | invented | notes |
|---|---|---|---|---|---|---|
| ab_ctrl | learned prior | 0.63 | 0.26 | 0.70 | 0.18 | all log-variances at the clamp (6) |
| **ab_anchor** | + `--anchor_prior 1e-6` | 0.69 | 0.20 | 0.77 | 0.12 | **frozen recipe**; scale pinned; best ELBO; 2 s: kept 0.90, sharp 0.79 |
| ab_fixedprior | `--fixed_prior` | 0.71 | 0.21 | 0.84 | 0.10 | ablation; 2 s: kept 0.91, sharp 0.85, lead 0.97 |
| ab_fp_res | fixed prior + `--residual` | 0.71 | 0.21 | 0.84 | 0.10 | identical: the residual (+≤1 logit) is cancelled; also a noise estimate (±0.01) |
| ab_res / ab_reenc | residual / reencode, learned prior | 0.73 / 0.69 | 0.18 / 0.24 | 0.76 / 0.70 | 0.16 / 0.20 | latent drift not slowed |
| ab_z1d / ab_anchor_z1d | 1-D z (32), per step | 0.78 / 0.72 | 0.16 / 0.17 | 0.74 / 0.68 | 0.18 / 0.21 | some cars drawn twice |
| ab_anchor_ol20_bo4 | 20% open-loop sequences, best-of-4 MSE | 0.78 | 0.19 | 0.51 | 0.36 | hedging: deterministic-like blur |
| ab_zseq64 | one 64-number z per sequence | 0.62 | 0.25 | 0.61 | 0.41 | cars smear, phantom traffic (closed-window mass 1.45) |
| cnn_det | deterministic | 0.84 | 0.14 | 0.54 | 0.32 | keeps cars by blurring |

Older runs (ws_*): open-loop training with plain MSE traded fading for blur (sharp 0.44); beta 1e-3 left z unused;
beta 1e-5 faded more.

## Reading

- Every variant ends between fading, blur and invented traffic at 4 s. Training the long uncorrected rollout
  directly (open loop, best-of-K, one z per sequence) brings blur or invention back, because a regression loss on
  a sample that cannot know the real future rewards the average.
- Prior-track KL growth (latent_eval.py) is not by itself evidence of drift: a perfect generator's sample also
  diverges from the one real future.
- With z's scale pinned, 70–80% of the KL sits in the ~14% of cells that contain road users; the spatial posterior
  is as wide as the prior (information travels as mean shifts).

## Is the learned prior's sigma_p an uncertainty map? (uncertainty_map.py, ab_anchor, 2 s, k = 8)

Per 6 m cell and step: sigma_p, KL(q||p), spread between 8 prior samples, error of their mean.

- sigma_p is not flat: 5th / 50th / 95th percentile 1.03 / 1.17 / 2.48; within one scene it varies by 31% of its mean.
- Every map is far larger in cells with a road user (error 134x, spread 26x, KL 19x, sigma_p 1.8x), so
  correlations over all cells partly measure "a road user is here".
- **Within road-user cells only** (the decisive test), Spearman: sigma_p vs spread 0.69, vs error 0.67. It predicts
  the model's error better than KL (0.56) and than the spread itself (0.59), from one forward pass and without the
  future. So sigma_p says which road users' futures are uncertain, not only where road users are.
- The fixed prior's maps (KL, spread, error) are about the same as the learned prior's; sample diversity and error
  at 2 s don't depend on the prior.
- Open: (1) a cell fully covered by a road user has more green than one it clips, which can inflate both sigma_p and
  the error; normalising by true green would rule this out. (2) The world map averaged over all cells mainly shows
  where traffic drives; averaging only over road-user cells would show where road users' futures are most uncertain.
- Claude and Fable 5.2 had both predicted a flat sigma_p (from the average widths); that prediction was wrong.

## Decision (Nov 11)

Frozen stage-2 recipe: learned prior + anchor (`--anchor_prior 1e-6`, beta 1e-4, `--pz_init_scale 0.01
--kl_warmup_steps 2000`, 20k steps), because its sigma_p is a validated uncertainty map. Planning and the encoder
comparison use 1 s and 2 s; 4 s is reported as a limitation with the ladder and closed-window numbers. The fixed
prior (`ab_fixedprior`) is kept as the ablation, including for the controller comparison.
