#!/bin/bash
# MAE-pretrain the vit encoder on single train frames (1xA100 40 GB, Cloud Bursting). Resumes automatically from
# runs/<RUN>/last.pt, so --requeue and the 20-minute idle shutdown only cost the minutes since the last checkpoint.
# Output: runs/<RUN>/encoder.pt, used by:  ENCODER=vit,RUN=vit_mae,INIT=runs/mae/encoder.pt train_wm.sbatch
#
#   sbatch scripts/hpc/pretrain_mae.sbatch
# Optional: STEPS=<n>   (sbatch --export=ALL,STEPS=30000 ...)
#SBATCH --job-name=mae
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

ARGS=""
if [ -n "${STEPS:-}" ]; then ARGS="$ARGS --steps $STEPS"; fi
python scripts/pretrain_mae.py --cache "$ROOT/round_cache" --run "${RUN:-mae}" --workers 10 $ARGS
