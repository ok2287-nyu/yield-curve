#!/bin/bash
# Train one world-model variant on 1xA100 40 GB (Cloud Bursting). Resumes automatically from runs/<RUN>/last.pt,
# so --requeue and the 20-minute idle shutdown only cost the minutes since the last checkpoint.
#
#   sbatch --export=ALL,ENCODER=cnn,RUN=cnn scripts/hpc/train_wm.sbatch
#   sbatch --export=ALL,ENCODER=vit,RUN=vit scripts/hpc/train_wm.sbatch
#   sbatch --export=ALL,ENCODER=vit,RUN=vit_mae,INIT=runs/mae/encoder.pt scripts/hpc/train_wm.sbatch
#   sbatch --export=ALL,ENCODER=cnn,RUN=cnn_nosg,EXTRA_ARGS=--no-prior_sg scripts/hpc/train_wm.sbatch   # ablation
# Optional: STEPS=<n> (decide after timing the first run; identical for all runs).
#SBATCH --job-name=wm
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out

set -euo pipefail
ROOT=/scratch/$USER/yield
source "$ROOT/venv/bin/activate"
cd "$ROOT/roundabout_wm"
nvidia-smi --query-gpu=name,memory.total --format=csv

ARGS="${EXTRA_ARGS:-}"
if [ -n "${INIT:-}" ]; then ARGS="$ARGS --init_encoder $INIT"; fi
if [ -n "${STEPS:-}" ]; then ARGS="$ARGS --steps $STEPS"; fi
python scripts/train_wm.py --cache "$ROOT/round_cache" --encoder "${ENCODER:-cnn}" --run "${RUN:-cnn}" \
  --workers 10 $ARGS
