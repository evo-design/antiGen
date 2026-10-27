#!/bin/bash
#SBATCH --job-name=eval_all_sc2
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_all_sc2_%j.out
#SBATCH --error=logs/eval_all_sc2_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# DATA_DIR / CKPT_DIR / OUTDIR are env overrides and EXTRA passes anything else
# through, so the 20-letter arm comes off this script rather than a fork. That arm
# names its inputs explicitly -- --alphabet, --train-examples, --test-examples,
# --root-counts, --variant-counts-dir -- and then DATA_DIR must be EMPTY: a published
# --data-dir beside those flags is a path read for nothing (NOTES s53).
DATA_DIR=${DATA_DIR-/home/ivan.specht/antiGen-data/output1}
CKPT_DIR=${CKPT_DIR:-/large_storage/hielab/ispecht/fitness-inference}
OUTDIR=${OUTDIR:-/large_storage/hielab/ispecht/fitness-inference/evals_sc2}
DATA_ARG=(); [ -n "$DATA_DIR" ] && DATA_ARG=(--data-dir "$DATA_DIR")
EXTRA=("$@")
echo "data-dir = ${DATA_DIR:-<none: named explicitly in EXTRA>}"
echo "ckpt-dir = $CKPT_DIR"
echo "outdir   = $OUTDIR"
echo "extra    = ${EXTRA[*]}"

antiGen-eval \
    --dataset sc2 \
    "${DATA_ARG[@]}" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUTDIR" \
    --priority-models blosum62,esm2 "${EXTRA[@]}"

echo "Job finished at: $(date)"
