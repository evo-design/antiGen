#!/bin/bash
#SBATCH --partition=gpu_batch
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --array=0-38
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_grid/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_grid/logs/%x_%A_%a.err

# GRID single-tree arm: ESM-2 {35m|650m} x {p4|lmhead}, one model per tree, 90/10 val,
# 3 SEEDS, recall@10 selection, early-stop patience 3.  SIZE/FT via env; set job-name at
# submit so %x logs disambiguate the cell.  array idx -> tree = PANEL[idx%13], seed = idx/13.
set -euo pipefail
: "${SIZE:?set SIZE=35m|650m}"; : "${FT:?set FT=p4|lmhead}"
case "$SIZE" in 35m) MODEL=facebook/esm2_t12_35M_UR50D ;; 650m) MODEL=facebook/esm2_t33_650M_UR50D ;; *) echo bad SIZE>&2; exit 1;; esac
case "$FT" in p4) LAYERS=4 ;; lmhead) LAYERS=0 ;; *) echo bad FT>&2; exit 1;; esac
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL=(denv1 denv2 denv3 denv4 h1n1pdm_ha h3n2_ha h5nx hmpv_a_g measles ncov rsv_a rsv_b vic_ha)
IDX=${SLURM_ARRAY_TASK_ID:-0}; TREE=${PANEL[$((IDX % 13))]}; SEED=$((IDX / 13))
F="$NA/examples_${TREE}@${D}_train.txt"; [ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }
OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_esm${SIZE}_${FT}_noattach/single_${TREE}_s${SEED}
mkdir -p "$OUT"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) SIZE=$SIZE FT=$FT tree=$TREE seed=$SEED"
"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name "$MODEL" --freeze-backbone --finetune-layers "$LAYERS" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best \
    --select-metric nll_pooled --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
