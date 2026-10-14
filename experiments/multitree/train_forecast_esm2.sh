#!/bin/bash
#SBATCH --job-name=forecast_esm2
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-32
#SBATCH --output=logs/forecast_esm2_%A_%a.out
#SBATCH --error=logs/forecast_esm2_%A_%a.err

# Causal forecasting arms at D = 2025-01-01. See FORECAST_DESIGN.md.
#
# Every arm trains ONLY on snapshot examples -- the builds as published at D -- so no
# post-D sequence, topology or ancestral reconstruction reaches training. Epochs and the
# arm are selected on a random split of those snapshot examples alone (--val-mode random),
# which is why the post-D test set is never passed to this script at all. It is opened
# once, afterwards, by score_forecast.py.
#
# Arms:
#   single_<tree>   one snapshot                 per-tree forecasting ability
#   joint_panel     all 9 forecastable snapshots does joint training improve forecasts
#   joint_all       those + 7 seasonal-flu trees does unrelated extra data help
#
# joint_all's extra trees are the retired seasonal-flu builds. They are delisted from
# Nextstrain and cannot be forecast-evaluated, but every one of their tips predates D
# (latest 2024.27 < 2025.00), so they are legitimate training-only context. That makes
# joint_all vs joint_panel the causal version of the joint5-vs-joint3 question.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
ODATA=$REPO/experiments/multitree/data
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/forecast}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4)
# Frozen, training-only. All tips <= 2024.27, verified before inclusion.
FROZEN=(h3n2 h1n1pdm h3n2_na h1n1pdm_na bvic_ha bvic_na byam_ha)

snap () { echo "$1=$FDATA/examples_$1@${D}_train.txt"; }
old  () { echo "$1=$ODATA/examples_$1_nocutoff.txt"; }

N_SEEDS=3
IDX=${SLURM_ARRAY_TASK_ID:-0}

EXAMPLES=()
if [ "$IDX" -lt 27 ]; then                      # 9 trees x 3 seeds
    TREE=${PANEL[$((IDX / N_SEEDS))]}
    SEED=$((IDX % N_SEEDS))
    CONFIG=single_${TREE}
    EXAMPLES+=("$(snap "$TREE")")
elif [ "$IDX" -lt 30 ]; then
    CONFIG=joint_panel; SEED=$((IDX - 27))
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(snap "$t")"); done
else
    CONFIG=joint_all;   SEED=$((IDX - 30))
    for t in "${PANEL[@]}";  do EXAMPLES+=("$(snap "$t")"); done
    for t in "${FROZEN[@]}"; do EXAMPLES+=("$(old  "$t")"); done
fi

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Forecast date: $D"
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
    --save-best --dump-cells \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
