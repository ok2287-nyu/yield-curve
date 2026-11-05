#!/bin/bash
# Two world-model runs side by side on ONE A100 (Cloud Bursting). A single run keeps the A100 only ~57% busy
# (ws_both_long, Nov 4), and A100 nodes are scarce, so each node we get trains two experiments. Each run resumes
# from runs/<RUN>/last.pt as usual (--requeue safe). The 12 CPUs are split: 5 data workers per run.
#
#   sbatch --export=ALL,ENCODER=cnn,STEPS=20000,WANDB=1,\
#   RUN1=ws_ol20,ARGS1="--init_from runs/cnn_det/best.pt --pz_init_scale 0.01 --kl_warmup_steps 2000 --beta 1e-4 --p_openloop 0.2",\
#   RUN2=ws_ol10,ARGS2="--init_from runs/cnn_det/best.pt --pz_init_scale 0.01 --kl_warmup_steps 2000 --beta 1e-4 --p_openloop 0.1" \
#   scripts/hpc/train_wm_pair.sbatch
#
# Each run writes its own log to runs/<RUN>/train.out; this job's .out has the GPU checks and the exit codes.
# Expect each run to be slower than alone (s/step in its log); the pair should still finish sooner than two jobs.
#SBATCH --job-name=wm2
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out

set -uo pipefail
ROOT=/scratch/$USER/yield
source "$ROOT/venv/bin/activate"
cd "$ROOT/roundabout_wm"
nvidia-smi --query-gpu=name,memory.total --format=csv

COMMON=""
if [ -n "${STEPS:-}" ]; then COMMON="$COMMON --steps $STEPS"; fi
if [ -n "${WANDB:-}" ]; then
  COMMON="$COMMON --wandb"
  if [ "$WANDB" = "offline" ]; then export WANDB_MODE=offline; fi
fi

launch() {   # $1 run name, $2 extra args
  mkdir -p "runs/$1"
  python -u scripts/train_wm.py --cache "$ROOT/round_cache" --encoder "${ENCODER:-cnn}" --run "$1" \
    --workers 5 $COMMON $2 >> "runs/$1/train.out" 2>&1 &
}
launch "$RUN1" "${ARGS1:-}"; P1=$!
launch "$RUN2" "${ARGS2:-}"; P2=$!
sleep 300; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv   # both running?
wait $P1; E1=$?
wait $P2; E2=$?
echo "$RUN1 exit $E1 | $RUN2 exit $E2"
exit $(( E1 || E2 ))
