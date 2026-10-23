#!/bin/bash
#SBATCH --job-name=fit_esm2_650m_partial
#SBATCH --partition=gpu_batch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=logs/fit_esm2_650m_partial_%j.out
#SBATCH --error=logs/fit_esm2_650m_partial_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# DATA_DIR / OUT_DIR are env overrides, and any extra arguments are passed straight
# through to antiGen-train, so a retrain on a rebuilt corpus (a new guide tree, say)
# reuses this exact recipe instead of a fork that would drift. The defaults reproduce
# the published checkpoints. The --msa/--metadata inputs, where present, are the
# open-data spike MSA and are NOT tree-dependent, so they are not parameterised.
DATA_DIR=${DATA_DIR:-/home/ivan.specht/antiGen-data/output1}
OUT_DIR=${OUT_DIR:-/large_storage/hielab/ispecht/fitness-inference}
EXTRA=("$@")
echo "data-dir = $DATA_DIR"
echo "out-dir  = $OUT_DIR"
echo "extra    = ${EXTRA[*]}"

antiGen-train \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t33_650M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --cutoff 1.0 --ignore-dead-ends \
    --epochs 32 --batch-size 4 --lr 0.0001 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" "${EXTRA[@]}"

echo "Job finished at: $(date)"
