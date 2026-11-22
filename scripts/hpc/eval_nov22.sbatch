#!/bin/bash
# Nov 22 evaluations, in order of urgency:
#   1. yield_starts with MC dropout (do the masks flag stopped road users the latent samples get confidently wrong?)
#   2. action_gain on the 4 s invariance runs (h20) and the DINO latent runs (higher beta)
#   3. 4 s saliency: inv4 vs h20
# Submit from roundabout_wm/:  sbatch scripts/hpc/eval_nov22.sbatch
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_nov22
#SBATCH --output=runs/eval_nov22_%j.out
source /scratch/ok2287/yield/venv/bin/activate && cd /scratch/ok2287/yield/roundabout_wm
export HF_HUB_OFFLINE=1
D="--cache ../round_cache --data_dir ../round_data"
mkdir -p runs/eval_nov22 runs/uncert_nov22/drop
echo "== yield_starts $(date)"
python scripts/yield_starts.py $D --ckpts runs/ab_anchor/last.pt runs/ab_inv4_01/last.pt \
  --n_windows 3000 --k 8 --masks 8 --out_dir runs/uncert_nov22/drop > runs/uncert_nov22/drop/starts.txt 2>&1
echo "== action_gain $(date)"
python scripts/action_gain.py $D --ckpts runs/ab_inv4_01/last.pt runs/ab_inv4_h20_01/last.pt \
  runs/ab_inv4_h20_02/last.pt runs/dino_z_b3e3/last.pt runs/dino_z_b1e2/last.pt > runs/eval_nov22/action_gain.txt 2>&1
echo "== saliency 4 s $(date)"
python scripts/saliency.py $D --ckpts runs/ab_inv4_01/last.pt runs/ab_inv4_h20_01/last.pt runs/ab_inv4_h20_02/last.pt \
  --csv runs/ab_inv4_01/action_gain.csv --T 20 --out_dir runs/saliency_h20_4s > runs/eval_nov22/saliency.txt 2>&1
echo "done $(date)"
