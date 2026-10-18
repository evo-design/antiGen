#!/bin/bash
#SBATCH --job-name=panel15
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-54
#SBATCH --output=logs/panel15_%A_%a.out
#SBATCH --error=logs/panel15_%A_%a.err

# Figure 5's arms on a FIFTEEN-tree panel: the eleven of s20 plus hMPV A/G, hMPV B/G,
# avian H9N2 HA and B/Victoria HA. D, recipe, seeds and split rules are unchanged, so
# the only difference from the eleven-tree figure is the panel.
#
# Where the four came from. `probe_forecast_panel.py` re-ran the wide-corpus probe at
# D = 2025-01-01 -- never done before; s18's probe was at the 2026 cutoff -- and then
# applied the five bars a forecast tree must clear beyond the wide-corpus rule (route A
# only, matching reference frames, a real time tree, a post-D budget, no tip overlap).
# Nine of 56 core datasets qualified; four of those nine were HA/NA or F/G cuts of the
# same strain sets, so one per pair was taken. These four are the HA/G members:
#
#   hmpv/a/G                   updated 2024-12-23   427 post-D targets,  79 novel
#   hmpv/b/G                   updated 2024-12-23   363 targets,         91 novel
#   avian-flu/h9n2/ha          updated 2024-12-06   354 targets,         46 novel
#   seasonal-flu/vic/ha/12y    updated 2024-12-29    74 targets,         14 novel
#
# All route A, all gene frames identical between snapshot and current build, and no
# pair among them shares tips (hmpv/a/G vs hmpv/b/G: 0 shared names). The panel goes
# 724 -> 1,918 post-D targets and 196 -> 426 novel.
#
# TWO DATA-QUALITY CAVEATS, recorded because they affect how the per-tree rows read:
#   * h9n2 HA drops 46.5% of its branch mutations to ambiguous `X` residues (7,665 of
#     16,498 in the snapshot). The kept mutations are real, but its targets are a
#     biased subsample of positions that happened to be called unambiguously.
#   * hmpv/a/G is 12.4% gap targets (53 of 427), far above anything else in the panel.
#     s16 established the ESM-2 arms put 0.00% of their mass on the gap column, so
#     those targets are dropped under `standard` and unwinnable under `common`.
#
# Arms:
#   tasks  0-7   joint_panel15  temporal, 15 snapshots, seeds 0-7  -> forecast15/
#   tasks  8-39  single_<new>   temporal, one snapshot, seeds 0-7  -> forecast/
#   tasks 40-42  joint15        random, 15 current builds, seeds 0-2 -> matched/
#   tasks 43-54  single_<new>   random, one current build, seeds 0-2 -> matched/
#
# The eleven existing single-tree controls are NOT retrained; only the joint arms and
# the four new trees' controls are new. Joint arms go to new roots so the nine- and
# eleven-tree runs behind the earlier figures stay untouched.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
ODATA=$REPO/experiments/multitree/data
FORECAST15=${1:-/large_storage/hielab/ispecht/multitree/forecast15}
FORECAST=${2:-/large_storage/hielab/ispecht/multitree/forecast}
MATCHED=${3:-/large_storage/hielab/ispecht/multitree/matched}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g h9n2_ha vic_ha)
NEW=(hmpv_a_g hmpv_b_g h9n2_ha vic_ha)

snap () { echo "$1=$FDATA/examples_$1@${D}_train.txt"; }
cur  () { echo "$1=$ODATA/examples_$1_nocutoff.txt"; }

IDX=${SLURM_ARRAY_TASK_ID:-0}
EXAMPLES=()
EXTRA=()

if [ "$IDX" -lt 8 ]; then                                  # temporal joint, seeds 0-7
    CONFIG=joint_panel15; SEED=$IDX; OUT_ROOT=$FORECAST15
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(snap "$t")"); done
elif [ "$IDX" -lt 40 ]; then                               # temporal singles, seeds 0-7
    TREE=${NEW[$(( (IDX - 8) / 8 ))]}; SEED=$(( (IDX - 8) % 8 ))
    CONFIG=single_${TREE}; OUT_ROOT=$FORECAST
    EXAMPLES+=("$(snap "$TREE")")
elif [ "$IDX" -lt 43 ]; then                               # random joint, seeds 0-2
    CONFIG=joint15; SEED=$((IDX - 40)); OUT_ROOT=$MATCHED
    EXTRA=(--test-frac 0.1)
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(cur "$t")"); done
else                                                       # random singles, seeds 0-2
    TREE=${NEW[$(( (IDX - 43) / 3 ))]}; SEED=$(( (IDX - 43) % 3 ))
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
