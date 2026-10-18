#!/bin/bash
#SBATCH --job-name=panel13
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-10
#SBATCH --output=logs/panel13_%A_%a.out
#SBATCH --error=logs/panel13_%A_%a.err

# The eleven-tree panel plus B/Victoria HA and hMPV B/G -- the two of s21's four
# additions that carry no known data-quality defect.
#
# Why this panel exists. At fifteen trees the de novo contrast against
# `where x which` fell from +0.112 [+0.031,+0.191] to +0.022 [-0.036,+0.081], i.e.
# back across zero, and joint-vs-single on all substitutions fell from +0.045 to
# +0.016. But two of the four trees added there are compromised:
#
#   h9n2_ha    drops 46.5% of its branch mutations to ambiguous `X` (7,665 of
#              16,498), so its post-D targets are a biased subsample of the
#              positions that happened to be called unambiguously
#   hmpv_a_g   is 12.4% gap targets against the panel's ~2%
#
# and together they are 42% of the fifteen-tree panel's targets. So the fifteen-tree
# result cannot distinguish "the claim is sensitive to harder trees" from "the claim
# is sensitive to lower-quality trees". This panel removes that ambiguity by adding
# only the two clean trees:
#
#   vic_ha     74 post-D targets, 14 novel; 0 ambiguous, 0 from-base mismatches
#   hmpv_b_g  363 post-D targets, 91 novel; 7.5% ambiguous, 3.6% gap
#
# NOTE hmpv_b_g is the weaker of the two on its own evidence -- at fifteen trees its
# per-tree joint-vs-single contrast was -0.020 [-0.045,+0.005] on all substitutions
# and -0.021 [-0.073,+0.000] on de novo, i.e. slightly negative both ways. It is kept
# because it is clean, not because it is favourable; dropping it on its result would
# be selecting the panel on the outcome.
#
# Only the JOINT arms are trained here. Single-tree controls do not depend on the
# panel beside them and already exist at the right seeds from the panel15 array:
# forecast/single_{vic_ha,hmpv_b_g}_s0-7 and matched/single_{...}_s0-2.
#
#   tasks 0-7   joint_panel13  temporal, 13 snapshots, seeds 0-7  -> forecast13/
#   tasks 8-10  joint13        random, 13 current builds, seeds 0-2 -> matched/

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
ODATA=$REPO/experiments/multitree/data
FORECAST13=${1:-/large_storage/hielab/ispecht/multitree/forecast13}
MATCHED=${2:-/large_storage/hielab/ispecht/multitree/matched}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_b_g vic_ha)

snap () { echo "$1=$FDATA/examples_$1@${D}_train.txt"; }
cur  () { echo "$1=$ODATA/examples_$1_nocutoff.txt"; }

IDX=${SLURM_ARRAY_TASK_ID:-0}
EXAMPLES=()
EXTRA=()

if [ "$IDX" -lt 8 ]; then                                  # temporal joint, seeds 0-7
    CONFIG=joint_panel13; SEED=$IDX; OUT_ROOT=$FORECAST13; TAG="forecast@${D}"
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(snap "$t")"); done
else                                                       # random joint, seeds 0-2
    CONFIG=joint13; SEED=$((IDX - 8)); OUT_ROOT=$MATCHED; TAG=multitree
    EXTRA=(--test-frac 0.1)
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(cur "$t")"); done
fi

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Output: $OUT"
echo "Training sources (${#EXAMPLES[@]}): ${EXAMPLES[*]}"

for spec in "${EXAMPLES[@]}"; do
    f=${spec#*=}
    [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

mkdir -p "$OUT"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 "${EXTRA[@]+"${EXTRA[@]}"}" \
    --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "$TAG" \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
