#!/bin/bash
# Closed-loop evaluation of the model planners (controller A) in ReplayEnv, 300 val windows, 6 s episodes.
#   sbatch scripts/hpc/planner_eval.sbatch                                  # log replay (others don't react)
#   sbatch --export=ALL,REACTIVE=1 scripts/hpc/planner_eval.sbatch          # reactive IDM referee
# Planners: mpc on ab_inv4_01 and ab_anchor (H1: does the leash make a controller cut in?), mpc_warp on ab_inv4_01
# (the untrained predict-then-warp baseline). Output: runs/planner_eval/{replay,react}.csv + .txt
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=planner_eval
#SBATCH --output=runs/planner_eval/slurm_%j.out
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
mkdir -p runs/planner_eval
TAG=replay; FLAG=""
if [ "${REACTIVE:-0}" = "1" ]; then TAG=react; FLAG="--reactive"; fi
echo "== $TAG $(date)"
python scripts/replay_eval.py --cache ../round_cache --data_dir ../round_data --n_windows 300 --horizon 30 $FLAG \
  --policies human,human_acc,brake,coast,go \
  --ckpts runs/ab_inv4_01/last.pt runs/ab_anchor/last.pt --planners mpc \
  --out runs/planner_eval/$TAG.csv > runs/planner_eval/$TAG.txt 2>&1
python scripts/replay_eval.py --cache ../round_cache --data_dir ../round_data --n_windows 300 --horizon 30 $FLAG \
  --policies human --ckpts runs/ab_inv4_01/last.pt --planners mpc_warp \
  --out runs/planner_eval/${TAG}_warp.csv > runs/planner_eval/${TAG}_warp.txt 2>&1
echo "done $(date)"
