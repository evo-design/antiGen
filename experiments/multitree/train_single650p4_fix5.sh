#!/bin/bash
#SBATCH --job-name=s650fix5
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-4
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs/%x_%A_%a.err

# Retrain the 5 single-650p4 runs whose metrics JSON was corrupted / that were left
# incompletely trained by the disk-full + SKIP-guard interaction. Recipe IDENTICAL to
# train_single650p4_add7.sh. idx -> (tree,seed) from PAIRS.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PAIRS=("h5nx 4" "h5nx 6" "h5nx 7" "h5nx 8" "h3n2_ha 8")
read -r TREE SEED <<< "${PAIRS[${SLURM_ARRAY_TASK_ID:-0}]}"
F="$NA/examples_${TREE}@${D}_train.txt"; [ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }
OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/single_${TREE}_s${SEED}
mkdir -p "$OUT"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) tree=$TREE seed=$SEED"
"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4 \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
