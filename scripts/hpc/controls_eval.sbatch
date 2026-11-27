#!/bin/bash
# Two controls / additions to the uncertainty fallback, both referees, 300 windows of SPLIT (default val):
#   1. risk-triggered fallback: the same fallback, triggered by the cheapest plan's imagined RISK instead of its
#      epistemic term, threshold matched per model to the uncertainty fallback's firing rate (epi > 0.1) on the val
#      decisions of runs/planner_eval/crash_unc_pp.csv (scripts/fallback_match.py). Thresholds are fixed on val and
#      reused for the test split.
#   2. headway guard (controller/headway.py) over the v4 / anchor planners and the v4 / anchor policies.
#   sbatch scripts/hpc/controls_eval.sbatch                                       # val
#   sbatch --export=ALL,SPLIT=test --job-name=ctl_test scripts/hpc/controls_eval.sbatch   # test, once
# Output: runs/planner_eval/ctl_{replay,react}.txt (val) or runs/final_eval/ctl_test_{replay,react}.txt + CSVs.
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --job-name=ctl_eval
#SBATCH --output=runs/planner_eval/ctl_slurm_%j.out
set -euo pipefail
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
export PYTHONUNBUFFERED=1
SPLIT=${SPLIT:-val}
D="--cache ../round_cache --data_dir ../round_data --split $SPLIT --n_windows 300 --horizon 30"
if [ "$SPLIT" = test ]; then O=runs/final_eval/ctl_test; mkdir -p runs/final_eval; else O=runs/planner_eval/ctl; fi
python scripts/fallback_match.py --csv runs/planner_eval/crash_unc_pp.csv --epi 0.1 | tee ${O}_thresholds.txt
thr() { awk -v m="$1" '$1 == m {print $3}' ${O}_thresholds.txt; }
for R in "" "--reactive"; do T=$([ -z "$R" ] && echo replay || echo react)
  : > ${O}_$T.txt
  for M in ab_inv4_01 ab_anchor; do
    TH=$(thr $M)
    echo "== $SPLIT $T $M risk fallback (threshold $TH, matched to epi > 0.1) $(date)" | tee -a ${O}_$T.txt
    python scripts/replay_eval.py $D $R --policies human --ckpts runs/$M/last.pt --planners mpc \
      --fallback_thresh "$TH" --fallback_on risk --out ${O}_${T}_risk_$M.csv >> ${O}_$T.txt 2>&1
  done
  echo "== $SPLIT $T headway guard (d0 1 m, tau 0.5 s, 4 s): planners and policies $(date)" | tee -a ${O}_$T.txt
  python scripts/replay_eval.py $D $R --policies human --ckpts runs/ab_inv4_01/last.pt runs/ab_anchor/last.pt \
    --planners mpc --policy_ckpts runs/pol_v4/best.pt runs/pol_anchor/best.pt --headway \
    --out ${O}_${T}_hw.csv >> ${O}_$T.txt 2>&1
done
echo "done $(date)"
