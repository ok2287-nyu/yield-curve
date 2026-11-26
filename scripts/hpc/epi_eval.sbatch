#!/bin/bash
# Planners with a strong, thresholded epistemic penalty (w_epi 10; threshold ~ the 90th / 95th percentile of the
# epistemic term in clean driving, from crash_uncertainty.py), closed loop on the same 300 val windows as
# runs/planner_eval/{replay,react}.csv, both referees. Output: runs/planner_eval/epi_{replay,react}.{csv,txt}
#   sbatch scripts/hpc/epi_eval.sbatch
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --job-name=epi_eval
#SBATCH --output=runs/planner_eval/epi_slurm_%j.out
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
D="--cache ../round_cache --data_dir ../round_data --n_windows 300 --horizon 30 --planners mpc --w_epi 10"
for R in "" "--reactive"; do T=$([ -z "$R" ] && echo replay || echo react)
  : > runs/planner_eval/epi_$T.txt
  for TH in 0.1 0.2; do
    echo "== $T thresh $TH $(date)" | tee -a runs/planner_eval/epi_$T.txt
    python scripts/replay_eval.py $D $R --epi_thresh $TH --policies human --ckpts runs/ab_inv4_01/last.pt \
      --out runs/planner_eval/epi_${T}_v4_t$TH.csv >> runs/planner_eval/epi_$T.txt 2>&1
  done
  echo "== $T anchor thresh 0.1 $(date)" | tee -a runs/planner_eval/epi_$T.txt
  python scripts/replay_eval.py $D $R --epi_thresh 0.1 --policies human --ckpts runs/ab_anchor/last.pt \
    --out runs/planner_eval/epi_${T}_anchor_t0.1.csv >> runs/planner_eval/epi_$T.txt 2>&1
done
echo "done $(date)"
