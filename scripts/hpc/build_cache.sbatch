#!/bin/bash
# Build the rendered cache for all Neuweiler recordings on NYU HPC Cloud Bursting (CPU work, no GPU used).
#
# Before submitting (on the burst login node):
#   sbatch scripts/hpc/build_cache.sbatch
# Runs on the CPU-only partition: GPU nodes with low GPU use for 20 min are shut down automatically.
# The build skips recordings already done, so if the job is cut short just submit it again.
#SBATCH --job-name=round_cache
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

ROOT=/scratch/$USER/yield
source "$ROOT/venv/bin/activate"                # the venv created in the setup steps
cd "$ROOT/roundabout_wm"
python tests/test_conventions.py
python scripts/build_cache.py --data_dir "$ROOT/round_data" --out_dir "$ROOT/round_cache" --skip_existing
python scripts/hpc/check_cache.py "$ROOT/round_cache"
