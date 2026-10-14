#!/bin/bash
#SBATCH --job-name=multitree_matched
#SBATCH --partition=gpu_batch
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --array=0-29
#SBATCH --output=logs/matched_esm2_%A_%a.out
#SBATCH --error=logs/matched_esm2_%A_%a.err

# Random-split arms on EXACTLY the nine trees the forecast panel uses, so the
# two evaluation regimes in Figure 5 are scored on the same datasets.
#
# THREE-way split (--test-frac 0.1): train 80% / val 10% / test 10%. The epoch is
# chosen on val and reported on test, so nothing in training or tuning has seen the
# numbers that get reported. The earlier two-way arms selected the epoch on val and
# then reported on that same val set, which is not a held-out estimate; those runs
# are superseded for any reported figure.
#
# The earlier decomp arm used five eval trees (ncov, h3n2, h1n1pdm, rsv_a,
# rsv_b); the forecast panel uses nine (ncov, rsv_a, rsv_b, h5nx, measles,
# denv1-4). Only three overlap, because seasonal flu is delisted from
# Nextstrain and cannot be forecast at all, while the forecast panel added
# trees that do carry dated snapshots. This arm closes that gap from the
# random-split side.
#
# Note what is NOT matched and cannot be: the random arm trains on today's
# full builds, the forecast arm on the build as published at D. That is the
# point of the comparison, not a confound to remove.
#
#   joint9          all nine trees, random split
#   single_<tree>   one tree, random split, all nine retrained here

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/matched}

ex () { echo "$1=$DATA/examples_$1_nocutoff.txt"; }

PANEL=(ncov rsv_a rsv_b h5nx measles denv1 denv2 denv3 denv4)
# All nine, not just the six that were missing: the ncov/rsv_a/rsv_b
# controls under decomp/ used the old two-way split, so their validation
# examples overlap this arm's test block and cannot be reused here.
NEW_SINGLES=(ncov rsv_a rsv_b h5nx measles denv1 denv2 denv3 denv4)

IDX=${SLURM_ARRAY_TASK_ID:-0}
EXAMPLES=()
if [ "$IDX" -lt 3 ]; then
    CONFIG=joint9; SEED=$IDX
    for t in "${PANEL[@]}"; do EXAMPLES+=("$(ex "$t")"); done
else
    T=$(( (IDX - 3) / 3 )); SEED=$(( (IDX - 3) % 3 ))
    TREE=${NEW_SINGLES[$T]}
    CONFIG=single_${TREE}
    EXAMPLES+=("$(ex "$TREE")")
fi

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Training sources (${#EXAMPLES[@]}): ${EXAMPLES[*]}"

mkdir -p "$OUT"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --test-frac 0.1 \
    --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment multitree \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
