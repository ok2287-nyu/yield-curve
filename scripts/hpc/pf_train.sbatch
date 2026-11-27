#!/bin/bash
# Pushforward run: EXACTLY ab_inv4_01's recipe (rebuilt from runs/ab_inv4_01/args.json) plus the pushforward loss.
# One change against the control ab_inv4_01. Resumes from runs/<RUN>/last.pt if requeued.
#   sbatch --export=ALL,RUN=ab_inv4_pf19,PF_K=19 scripts/hpc/pf_train.sbatch
#   sbatch --export=ALL,RUN=ab_inv4_pf10,PF_K=10 scripts/hpc/pf_train.sbatch
# Optional: PF_W (loss weight, default 1), PF_RAMP (curriculum steps, default 10000), REF (default ab_inv4_01).
#SBATCH --job-name=pf_train
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=runs/pf_train_%x_%j.out
set -euo pipefail
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
export PYTHONUNBUFFERED=1
REF=${REF:-ab_inv4_01}; RUN=${RUN:?set RUN}; PF_K=${PF_K:-19}; PF_W=${PF_W:-1}; PF_RAMP=${PF_RAMP:-10000}
RECIPE=$(python - "runs/$REF/args.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
skip = {"run", "runs_dir", "cache", "workers", "wandb", "wandb_project", "wandb_entity", "max_windows"}
neg = {"prior_sg", "teacher_forcing"}                       # BooleanOptionalAction: --x / --no-x
out = []
for k, v in a.items():
    if k in skip or k.startswith("pf_") or v is None:
        continue
    if k in neg:
        out.append(f"--{k}" if v else f"--no-{k}")
    elif isinstance(v, bool):
        if v:
            out.append(f"--{k}")
    else:
        out.append(f"--{k}={v}")                       # = form: values like -3,-1.5 are not flags
print(" ".join(out))
EOF
)
echo "recipe from $REF: $RECIPE"
echo "pushforward: --pf_weight $PF_W --pf_k_max $PF_K --pf_ramp_steps $PF_RAMP"
python -u scripts/train_wm.py --cache ../round_cache --run "$RUN" --workers 10 $RECIPE \
  --pf_weight "$PF_W" --pf_k_max "$PF_K" --pf_ramp_steps "$PF_RAMP"
