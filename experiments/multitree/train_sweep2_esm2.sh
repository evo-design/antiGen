#!/bin/bash
#SBATCH --job-name=multitree_scale
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-11
#SBATCH --output=logs/scale_esm2_%A_%a.out
#SBATCH --error=logs/scale_esm2_%A_%a.err

# Does the joint-training benefit grow with the number of trees?
#
# Every arm contains the same five EVALUATION trees (ncov, h3n2, h1n1pdm, rsv_a, rsv_b)
# and differs only in how many additional trees are trained alongside them, so the
# held-out sets are identical across arms and the comparison is paired:
#
#   joint5   (from the first sweep)  the 5 evaluation trees alone
#   joint11  +6 extra                one HA, one NA, one flavivirus, three unrelated
#   joint17  +12 extra
#   joint23  +18 extra               every usable tree
#
# The tiers are nested and deliberately balanced across virus families, so a trend cannot
# be an artefact of adding (say) six more influenza trees at one step.
#
# hafam is the mechanistic control: the six HA trees only, evaluated on h3n2 and h1n1pdm.
# The first sweep found the benefit concentrated on those two same-protein trees, so if
# relatedness is what matters, hafam should reproduce most of the gain without ncov or RSV.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/sweep}

ex () { echo "$1=$DATA/examples_$1_nocutoff.txt"; }

EVAL_TREES=(ncov h3n2 h1n1pdm rsv_a rsv_b)
TIER1=(bvic_ha h3n2_na denv1 measles evd68 rabies)
TIER2=(byam_ha h1n1pdm_na denv2 mumps h5n1 wnv)
TIER3=(bvic_na h5nx denv3 denv4 zika mpox)
HA_FAMILY=(h3n2 h1n1pdm bvic_ha byam_ha h5n1 h5nx)

CONFIGS=(joint11 joint17 joint23 hafam)
N_SEEDS=3

IDX=${SLURM_ARRAY_TASK_ID:-0}
CONFIG=${CONFIGS[$((IDX / N_SEEDS))]}
SEED=$((IDX % N_SEEDS))

case "$CONFIG" in
    joint11) TREES=("${EVAL_TREES[@]}" "${TIER1[@]}") ;;
    joint17) TREES=("${EVAL_TREES[@]}" "${TIER1[@]}" "${TIER2[@]}") ;;
    joint23) TREES=("${EVAL_TREES[@]}" "${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}") ;;
    hafam)   TREES=("${HA_FAMILY[@]}") ;;
    *) echo "Unknown config: $CONFIG" >&2; exit 1 ;;
esac

EXAMPLES=()
for tree in "${TREES[@]}"; do EXAMPLES+=("$(ex "$tree")"); done

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Trees (${#TREES[@]}): ${TREES[*]}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

mkdir -p "$OUT"

# run_train.py == antiGen-train plus a workaround for this environment's broken
# torchvision; see its docstring. Swap back to `antiGen-train` once that is fixed.
python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
