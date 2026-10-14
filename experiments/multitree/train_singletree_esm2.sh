#!/bin/bash
#SBATCH --job-name=singletree_esm2
#SBATCH --partition=gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=logs/singletree_esm2_%A_%a.out
#SBATCH --error=logs/singletree_esm2_%A_%a.err

# Control runs for the multi-tree experiment: the same recipe as
# train_multitree_esm2.sh, but one tree per job. Only if the joint model beats these
# per-tree models is anything actually transferring between trees.
#
# The source NAME and --split-seed are kept identical to the joint run, so
# split_examples() draws each tree's held-out set from the same per-source RNG stream
# (seeded with [split_seed, crc32(name)]) and the validation sets match example for
# example. Do not rename the sources here.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/single}

NAMES=(ncov h3n2 h1n1pdm)
FILES=(examples_ncov_S_nocutoff.txt examples_h3n2_HA_nocutoff.txt examples_h1n1pdm_HA_nocutoff.txt)

IDX=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$IDX]}
FILE=${FILES[$IDX]}
OUT=$OUT_ROOT/$NAME

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${NAME})"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

mkdir -p "$OUT"

# run_train.py == antiGen-train plus a workaround for this environment's broken
# torchvision; see its docstring. Swap back to `antiGen-train` once that is fixed.
python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "$NAME=$DATA/$FILE" \
    --val-mode random --val-frac 0.1 --split-seed 0 \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
