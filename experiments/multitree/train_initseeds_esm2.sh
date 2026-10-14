#!/bin/bash
#SBATCH --job-name=fcst_init
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-3
#SBATCH --output=logs/fcst_init_%A_%a.out
#SBATCH --error=logs/fcst_init_%A_%a.err

# joint_panel replicates that share ONE validation split (--split-seed 0) and differ
# only in initialisation (--seed 100..103).
#
# Why this arm exists. Every existing seed uses --split-seed == --seed, so each model
# holds out a DIFFERENT 10%. An ensemble of them has therefore trained on every
# pre-D example, and no pre-D validation set is clean for the ensemble -- which is
# exactly the object we now want to tune a gate against. Fixing the split and varying
# only the init gives an ensemble whose held-out split is clean for all its members,
# so a gate can be selected on pre-D data the ensemble never saw and then applied,
# unchanged, to the post-D set. Without this the gate must be selected on single-seed
# maps and applied to an ensemble -- a mismatch that is a plausible reason the first
# weight search found nothing.
#
# These go in a SEPARATE root so they cannot be confused with the s0-s7 series, whose
# members are not interchangeable with them.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/forecast_fixedsplit}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4)
SEED=$((100 + ${SLURM_ARRAY_TASK_ID:-0}))
SPLIT=0

EXAMPLES=()
for t in "${PANEL[@]}"; do EXAMPLES+=("$t=$FDATA/examples_$t@${D}_train.txt"); done

OUT=$OUT_ROOT/joint_panel_s${SEED}
mkdir -p "$OUT"
echo "joint_panel seed ${SEED} -> $OUT  ($(date), $(hostname))"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SPLIT" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "done $(date)"
