#!/bin/bash
#SBATCH --job-name=multitree_decomp
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-20
#SBATCH --output=logs/decomp_esm2_%A_%a.out
#SBATCH --error=logs/decomp_esm2_%A_%a.err

# Re-runs the arms the decomposition analysis needs, with the two flags the earlier
# sweeps lacked:
#
#   --save-best   the earlier sweeps kept only the epoch-19 weights, which are well past
#                 the best epoch (typically 1-10), so every saved checkpoint was the
#                 overfit model. Phase 3's ablations need the scored model.
#   --dump-cells  writes the full per-cell validation probability maps at the best epoch,
#                 which is what the where/which decomposition reads.
#
# Both runs also pick up the two new baseline rungs (where_only, where_times_which) that
# did not exist when the scaling sweep ran.
#
# Arms: joint5 and the five single-tree controls (the one solid effect, -0.125, is
# joint5 vs single, so that pairing is what Phase 2 decomposes), plus joint23 to get the
# ladder on all 23 trees.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/decomp}

ex () { echo "$1=$DATA/examples_$1_nocutoff.txt"; }

EVAL_TREES=(ncov h3n2 h1n1pdm rsv_a rsv_b)
ALL_TREES=(ncov h3n2 h1n1pdm rsv_a rsv_b bvic_ha h3n2_na denv1 measles evd68 rabies
           byam_ha h1n1pdm_na denv2 mumps h5n1 wnv bvic_na h5nx denv3 denv4 zika mpox)

# 0-2 joint5 | 3-17 singles (5 trees x 3 seeds) | 18-20 joint23
IDX=${SLURM_ARRAY_TASK_ID:-0}
if   [ "$IDX" -lt 3 ];  then CONFIG=joint5;  SEED=$IDX;              TREES=("${EVAL_TREES[@]}")
elif [ "$IDX" -lt 18 ]; then
    T=$(( (IDX - 3) / 3 )); SEED=$(( (IDX - 3) % 3 ))
    CONFIG=single_${EVAL_TREES[$T]}; TREES=("${EVAL_TREES[$T]}")
else CONFIG=joint23; SEED=$(( IDX - 18 )); TREES=("${ALL_TREES[@]}")
fi

EXAMPLES=()
for tree in "${TREES[@]}"; do EXAMPLES+=("$(ex "$tree")"); done

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Trees (${#TREES[@]}): ${TREES[*]}"

mkdir -p "$OUT"

# run_train.py == antiGen-train plus a workaround for this environment's broken
# torchvision; see its docstring.
python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --dump-cells \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
