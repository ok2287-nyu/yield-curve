#!/bin/bash
# Train a learned policy (controller B) through a frozen world model.
#   sbatch --export=ALL,NAME=pol_v4,WM=runs/ab_inv4_01/last.pt,LAM=0.5 scripts/hpc/train_policy.sbatch
#   sbatch --export=ALL,NAME=pol_v4_nou,WM=runs/ab_inv4_01/last.pt,LAM=0 scripts/hpc/train_policy.sbatch
#   sbatch --export=ALL,NAME=pol_anchor,WM=runs/ab_anchor/last.pt,LAM=0.5 scripts/hpc/train_policy.sbatch
# Extra flags in EXTRA_ARGS (e.g. EXTRA_ARGS="--steps 20000").
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --job-name=policy
#SBATCH --output=runs/policy_%j.out
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
: "${NAME:?set NAME}" "${WM:?set WM}"
echo "== $NAME wm=$WM lam=${LAM:-0.5} $(date)"
python scripts/train_policy.py --cache ../round_cache --wm "$WM" --name "$NAME" --lam "${LAM:-0.5}" \
  --steps 10000 --workers 10 $EXTRA_ARGS
echo "done $(date)"
