#!/bin/bash
# Planner fallback: when the cheapest plan's epistemic term exceeds 0.1 (~ clean-driving p90), take the plan with the
# lowest imagined risk instead. v4 and anchor, both referees, the same 300 val windows as the earlier planner runs.
# Output: runs/planner_eval/fb_{replay,react}.txt + per-run CSVs.   sbatch scripts/hpc/fallback_eval.sbatch
# TEST recordings (run once): sbatch --export=ALL,SPLIT=test --job-name=fb_test scripts/hpc/fallback_eval.sbatch
#   -> runs/final_eval/fb_test_{replay,react}.txt + CSVs
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=fb_eval
#SBATCH --output=runs/planner_eval/fb_slurm_%j.out
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
SPLIT=${SPLIT:-val}
D="--cache ../round_cache --data_dir ../round_data --n_windows 300 --horizon 30 --planners mpc --fallback_thresh 0.1"
if [ "$SPLIT" = test ]; then D="$D --split test"; O=runs/final_eval/fb_test; mkdir -p runs/final_eval; else O=runs/planner_eval/fb; fi
for R in "" "--reactive"; do T=$([ -z "$R" ] && echo replay || echo react)
  : > ${O}_$T.txt
  for M in ab_inv4_01 ab_anchor; do
    echo "== $SPLIT $T $M fallback 0.1 $(date)" | tee -a ${O}_$T.txt
    python scripts/replay_eval.py $D $R --policies human --ckpts runs/$M/last.pt \
      --out ${O}_${T}_$M.csv >> ${O}_$T.txt 2>&1
  done
done
echo "done $(date)"
