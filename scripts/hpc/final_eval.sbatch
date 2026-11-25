#!/bin/bash
# Final closed-loop numbers on the TEST recordings (run once), plus episode sheets from the val runs.
#   sbatch scripts/hpc/final_eval.sbatch                       # everything
#   sbatch --export=ALL,STEPS="sheets" scripts/hpc/final_eval.sbatch
# Outputs: runs/final_eval/test_{replay,react}{,_warp}.{csv,txt}; runs/planner_eval/sheets/*.png (rounD imagery:
# keep on HPC / your machine).
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --job-name=final_eval
#SBATCH --output=runs/final_eval_%j.out
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
D="--cache ../round_cache --data_dir ../round_data"
STEPS=${STEPS:-"sheets test"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p runs/final_eval runs/planner_eval/sheets
if has sheets; then
  echo "== sheets $(date)"
  S="python scripts/episode_sheet.py $D --n 8"
  $S --csv runs/planner_eval/pol_replay.csv --controller pol:pol_v4 --policy_ckpts runs/pol_v4/best.pt \
     --select crash --out runs/planner_eval/sheets/pol_v4_crash.png
  $S --csv runs/planner_eval/pol_replay.csv --controller pol:pol_v4 --policy_ckpts runs/pol_v4/best.pt \
     --select differ --out runs/planner_eval/sheets/pol_v4_differ.png
  $S --csv runs/planner_eval/replay.csv --controller mpc:ab_inv4_01 --ckpts runs/ab_inv4_01/last.pt --planners mpc \
     --select crash --out runs/planner_eval/sheets/mpc_v4_crash.png
  $S --csv runs/planner_eval/pol_replay.csv --controller pol:pol_anchor --policy_ckpts runs/pol_anchor/best.pt \
     --select crash --out runs/planner_eval/sheets/pol_anchor_crash.png
fi
if has test; then
  for R in "" "--reactive"; do T=$([ -z "$R" ] && echo replay || echo react)
    echo "== test $T $(date)"
    python scripts/replay_eval.py $D --split test --n_windows 300 --horizon 30 $R \
      --policies human,human_acc,brake,coast,go --ckpts runs/ab_inv4_01/last.pt runs/ab_anchor/last.pt --planners mpc \
      --policy_ckpts runs/pol_v4/best.pt runs/pol_v4_nou/best.pt runs/pol_anchor/best.pt \
      --out runs/final_eval/test_$T.csv > runs/final_eval/test_$T.txt 2>&1
    python scripts/replay_eval.py $D --split test --n_windows 300 --horizon 30 $R \
      --policies human --ckpts runs/ab_inv4_01/last.pt --planners mpc_warp \
      --out runs/final_eval/test_${T}_warp.csv > runs/final_eval/test_${T}_warp.txt 2>&1
  done
fi
echo "done $(date)"
