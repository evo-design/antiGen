#!/bin/bash
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_grid/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_grid/logs/%x_%A_%a.err

# GRID joint arm: ESM-2 {35m|650m} x {p4|lmhead}, ARM in {joint13|joint49}, 90/10 val,
# 3 SEEDS (array idx = seed), macro val recall@10 selection, early-stop patience 3.
set -euo pipefail
: "${SIZE:?set SIZE=35m|650m}"; : "${FT:?set FT=p4|lmhead}"; : "${ARM:?set ARM=joint13|joint49}"
case "$SIZE" in 35m) MODEL=facebook/esm2_t12_35M_UR50D ;; 650m) MODEL=facebook/esm2_t33_650M_UR50D ;; *) echo bad SIZE>&2; exit 1;; esac
case "$FT" in p4) LAYERS=4 ;; lmhead) LAYERS=0 ;; *) echo bad FT>&2; exit 1;; esac
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
NA=$HERE/data/forecast_noattach
D=2025-01-01
SEED=${SLURM_ARRAY_TASK_ID:-0}
PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)
case "$ARM" in
  joint13) EXAMPLES=(); for t in "${PANEL[@]}"; do f="$NA/examples_${t}@${D}_train.txt"; [ -f "$f" ] || { echo "MISSING $f">&2; exit 1; }; EXAMPLES+=("${t}=${f}"); done ;;
  joint49) read -r -a EXAMPLES <<< "$($PY "$HERE/fc13_plus_core_specs.py" --mode surface --route snapshot)" ;;
  *) echo bad ARM>&2; exit 1 ;;
esac
echo "ARM=$ARM datasets=${#EXAMPLES[@]}"
OUT=/large_storage/hielab/aakarsh/multitree/fc13_grid/${ARM}_esm${SIZE}_${FT}_val_s${SEED}
mkdir -p "$OUT"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) SIZE=$SIZE FT=$FT ARM=$ARM seed=$SEED"
"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name "$MODEL" --freeze-backbone --finetune-layers "$LAYERS" \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source \
    --select-metric nll_pooled --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
