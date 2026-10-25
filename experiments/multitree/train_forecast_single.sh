#!/bin/bash
#SBATCH --job-name=fcSingle
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --array=0-13
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_single_650m/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_single_650m/logs/%x_%A_%a.err

# Per-virus single-tree controls on the D=2025-01-01 snapshot sets, over the joint-14 panel
# (15-tree panel minus h9n2_ha), seed 0. One array task per tree.
#
#   arg 1  FAMILY = esm650 | xformer
#     esm650  : ESM-2 650M, LM-head only (--freeze-backbone --finetune-layers 0) -- the
#               per-virus control for the joint-650M arms.
#     xformer : from-scratch transformer (d_model 512, 4 layers), no pretraining -- the
#               architecture control. Per-tree only; transformer is not length-agnostic.
#
# Same split/selection recipe as the joint arms (val random 0.1, split-seed 0, save-best,
# E=22 alphabet, forecast@D tag, lr 1e-4 / 20 ep / batch 8). Runs Ivan's feature-complete
# antiGen via run_train_fc.py; output -> my storage, one dir per (family, tree).

set -euo pipefail

PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
# Vendored in-repo antiGen; selection = macro recall@10 via --select-metric; gaps/stops masked.
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
D=2025-01-01
FAMILY=${1:?usage: sbatch train_forecast_single.sh <esm650|xformer>}

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g vic_ha)
TREE=${PANEL[${SLURM_ARRAY_TASK_ID:-0}]}
F="$FDATA/examples_${TREE}@${D}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

case "$FAMILY" in
  esm650)
    OUT_ROOT=/large_storage/hielab/aakarsh/multitree/forecast_single_650m
    MODEL_ARGS=(--model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-'
                --esm2-model-name facebook/esm2_t33_650M_UR50D
                --freeze-backbone --finetune-layers 0) ;;
  xformer)
    OUT_ROOT=/large_storage/hielab/aakarsh/multitree/forecast_single_xformer
    MODEL_ARGS=(--model-type transformer --alphabet 'ACDEFGHIKLMNPQRSTVWY*-'
                --d-model 512 --num-layers 4) ;;
  *) echo "unknown FAMILY '$FAMILY' (want esm650|xformer)" >&2; exit 1 ;;
esac

OUT="$OUT_ROOT/single_${TREE}_s0"
mkdir -p "$OUT" "$OUT_ROOT/logs"

echo "Job started: $(date)  node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-0}"
echo "family=$FAMILY  tree=$TREE  out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" "$LAUNCH" \
    "${MODEL_ARGS[@]}" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 \
    --split-seed 0 --seed 42 \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --dump-cells \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "Job finished: $(date)"
