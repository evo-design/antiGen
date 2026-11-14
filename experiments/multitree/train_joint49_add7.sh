#!/bin/bash
#SBATCH --job-name=j49add7
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-6
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%A_%a.err

# +7 SEEDS (s3..s9) for the 49-tree ESM-2 650M PARTIAL-4 joint arm, to grow the existing
# 3-seed ensemble to 10. Recipe IDENTICAL to train_fc13_joint49_p4_val_seeds.sh (49-tree
# corpus = 13 fc13 + 36 snapshot surface; 90/10 val, 30 ep, save-best + save-every-epoch +
# early-stop 3, recall10_macro select). array idx = 0..6 -> seed = 3 + idx.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
D=2025-01-01
SEED=$((3 + ${SLURM_ARRAY_TASK_ID:-0}))
read -r -a EXAMPLES <<< "$($PY "$HERE/fc13_plus_core_specs.py" --mode surface --route snapshot)"
echo "datasets=${#EXAMPLES[@]} (expect 49)  seed=$SEED"
OUT=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/joint_surface_snapshot_p4_val_s${SEED}
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) seed=$SEED (49-tree p4 add7)"
"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
