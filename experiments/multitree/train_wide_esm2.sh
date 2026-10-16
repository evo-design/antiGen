#!/bin/bash
#SBATCH --job-name=wide_esm2
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-5
#SBATCH --output=logs/wide_esm2_%A_%a.out
#SBATCH --error=logs/wide_esm2_%A_%a.err

# The wide pre-cutoff corpus at D = 2026-05-01. See NOTES.md s18.
#
# Every tree here carries no information observed after the cutoff, either because its
# build was published before it (route A, 40 trees) or because every one of its tips
# predates it (route B, 3 trees). The evaluation is a single tree observed wholly after
# the cutoff, supplied separately; it is not passed to this script and never reaches
# training or epoch selection. Epochs are chosen on a random split of the corpus alone
# (--val-mode random), exactly as the forecast arms chose theirs.
#
# Arms, 3 seeds each:
#   joint_wide    all 43 corpus trees        does a 7.5x larger pre-cutoff corpus help
#   joint_panel   the 9 panel trees only     the control it has to beat
#
# joint_panel is the same nine datasets as the old forecast panel, taken at this cutoff
# rather than 2025-01-01, so the contrast isolates the extra 34 trees and nothing else:
# same recipe, same seeds, same split rule, same evaluation.
#
# Recipe matches train_forecast_esm2.sh exactly -- same backbone, same 4 unfrozen
# layers, same lr, batch, epochs, split rule and seeds. The corpus is 5x larger, so 20
# epochs is 5x the work (expect hours, not minutes), but --save-best picks the epoch, so
# the count only needs to be generous enough to contain the minimum.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
WDATA=$REPO/experiments/multitree/data/wide
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/wide_train}
D=2026-05-01
MANIFEST=$WDATA/corpus@${D}.json

# The nine panel datasets, by their short names in the manifest.
PANEL=(ncov rsv_a_f rsv_b_f measles_h denv1 denv2 denv3 denv4 h5nx_ha)

N_SEEDS=3
IDX=${SLURM_ARRAY_TASK_ID:-0}

if [ "$IDX" -lt "$N_SEEDS" ]; then
    CONFIG=joint_wide; SEED=$IDX
    # Read the trees from the manifest rather than a hardcoded list, so the arm cannot
    # silently disagree with the corpus that was generated and validated.
    mapfile -t SHORTS < <(python3 -c "
import json; print('\n'.join(sorted(json.load(open('$MANIFEST')))))")
else
    CONFIG=joint_panel; SEED=$((IDX - N_SEEDS))
    SHORTS=("${PANEL[@]}")
fi

EXAMPLES=()
for s in "${SHORTS[@]}"; do
    path=$(python3 -c "
import json; print(json.load(open('$MANIFEST'))['$s']['path'])")
    [ -f "$path" ] || { echo "missing examples file for $s: $path" >&2; exit 1; }
    EXAMPLES+=("$s=$path")
done

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Cutoff: $D"
echo "Training sources (${#EXAMPLES[@]}): ${EXAMPLES[*]}"

mkdir -p "$OUT"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "wide@${D}" \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
