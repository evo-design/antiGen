#!/bin/bash
#SBATCH --job-name=xffix5
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_val_noattach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_val_noattach/logs/%x_%A_%a.err

# Retrain the 5 from-scratch transformer runs whose best.pt was truncated by the disk-full
# event. Recipe IDENTICAL to train_xformer_add7.sh. idx -> (tree,seed) from PAIRS.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PAIRS=("rsv_a 8" "rsv_a 9" "rsv_b 9" "measles 3" "measles 9")
read -r TREE SEED <<< "${PAIRS[${SLURM_ARRAY_TASK_ID:-0}]}"
F="$NA/examples_${TREE}@${D}_train.txt"; [ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }
OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_val_noattach/single_${TREE}_s${SEED}
mkdir -p "$OUT"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) tree=$TREE seed=$SEED"
"$PY" "$LAUNCH" \
    --model-type transformer --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' --d-model 512 --num-layers 4 \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 60 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best \
    --select-metric nll_pooled --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
