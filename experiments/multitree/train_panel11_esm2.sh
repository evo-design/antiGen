#!/bin/bash
#SBATCH --job-name=panel11
#SBATCH --partition=gpu_high_mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-32
#SBATCH --output=logs/panel11_%A_%a.out
#SBATCH --error=logs/panel11_%A_%a.err

# Figure 5's arms re-run on an ELEVEN-tree panel: the nine of s11 plus seasonal flu
# H3N2 HA and H1N1pdm HA. Everything else -- D, recipe, seeds, split rules, scoring
# -- is unchanged, so the only difference from the published figure is the panel.
#
# Why the two flu trees are now available. NOTES s8 recorded "seasonal flu has no
# versions at any date probed" and "zero of Charon's 318 datasets match flu/seasonal".
# That was a prefix mismatch (the paths are `seasonal-flu/...`, not `flu/seasonal/...`),
# exactly as s17 suspected. Both HA builds are archived at D and verified by route A:
#
#   seasonal-flu/h3n2/ha/12y@2025-01-01     updated 2024-12-29, 1576 tips, latest 2024.95
#   seasonal-flu/h1n1pdm/ha/12y@2025-01-01  updated 2024-12-29, 1470 tips, latest 2024.96
#
# `meta.updated` precedes D and every tip predates D, so neither can carry anything
# observed after the cutoff. Their gene frames (SigPep/HA1/HA2) are identical between
# the snapshot and today's build, which is what make_forecast_testset.py requires, and
# neither shares a single tip name with the other or with h5nx HA.
#
# Post-D budget: h3n2_ha 95 targets (22 novel), h1n1pdm_ha 65 (18). The panel goes
# 564 -> 724 targets and 156 -> 196 novel, and h5nx drops from 41% to 32% of the test
# set, which is the top-heaviness s11 flagged.
#
# Arms:
#   tasks  0-7   joint_panel11  temporal, 11 snapshots, seeds 0-7  -> forecast11/
#   tasks  8-23  single_<flu>   temporal, one snapshot, seeds 0-7  -> forecast/
#   tasks 24-26  joint11        random, 11 current builds, seeds 0-2 -> matched/
#   tasks 27-32  single_<flu>   random, one current build, seeds 0-2 -> matched/
#
# The nine existing single-tree controls are NOT retrained: a single-tree arm does not
# depend on the panel, and its files are already in forecast/ (seeds 0-7) and matched/
# (seeds 0-2). Only the joint arms and the two new trees' controls are new. The joint
# arms go to NEW roots (forecast11/, matched/joint11_*) so the nine-tree runs behind
# the published figure stay untouched.
#
# Seeds: 8 in the temporal regime, which is the matched K the figure reports (s15/s16
# -- joint and single must ensemble at the SAME K). 3 in the random regime, which
# cannot ensemble at all because each seed carries its own three-way split.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
ODATA=$REPO/experiments/multitree/data
FORECAST11=${1:-/large_storage/hielab/ispecht/multitree/forecast11}
FORECAST=${2:-/large_storage/hielab/ispecht/multitree/forecast}
MATCHED=${3:-/large_storage/hielab/ispecht/multitree/matched}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha)
NEW=(h3n2_ha h1n1pdm_ha)

snap () { echo "$1=$FDATA/examples_$1@${D}_train.txt"; }
cur  () { echo "$1=$ODATA/examples_$1_nocutoff.txt"; }

IDX=${SLURM_ARRAY_TASK_ID:-0}
EXAMPLES=()
EXTRA=()

if [ "$IDX" -lt 8 ]; then                                  # temporal joint, seeds 0-7
    CONFIG=joint_panel11; SEED=$IDX; OUT_ROOT=$FORECAST11
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(snap "$t")"); done
elif [ "$IDX" -lt 24 ]; then                               # temporal singles, seeds 0-7
    TREE=${NEW[$(( (IDX - 8) / 8 ))]}; SEED=$(( (IDX - 8) % 8 ))
    CONFIG=single_${TREE}; OUT_ROOT=$FORECAST
    EXAMPLES+=("$(snap "$TREE")")
elif [ "$IDX" -lt 27 ]; then                               # random joint, seeds 0-2
    CONFIG=joint11; SEED=$((IDX - 24)); OUT_ROOT=$MATCHED
    EXTRA=(--test-frac 0.1)
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(cur "$t")"); done
else                                                       # random singles, seeds 0-2
    TREE=${NEW[$(( (IDX - 27) / 3 ))]}; SEED=$(( (IDX - 27) % 3 ))
    CONFIG=single_${TREE}; OUT_ROOT=$MATCHED
    EXTRA=(--test-frac 0.1)
    EXAMPLES+=("$(cur "$TREE")")
fi

# The temporal regime tags its runs forecast@D and the random one `multitree`, matching
# train_forecast_esm2.sh and train_matched_esm2.sh so antiGen-eval keeps skipping both
# unless given --include-experimental.
if [ "$OUT_ROOT" = "$MATCHED" ]; then TAG=multitree; else TAG="forecast@${D}"; fi

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
