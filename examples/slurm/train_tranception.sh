#!/bin/bash
#SBATCH --job-name=train_tranception
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_tranception_%j.out
#SBATCH --error=logs/train_tranception_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

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
    --model-type tranception \
    --msa /home/ivan.specht/sc2-open-data/spike.fasta \
    --metadata /home/ivan.specht/sc2-open-data/metadata.tsv \
    --cutoff 1.0 --ignore-dead-ends \
    --lr 0.0001 --epochs 32 \
    --batch-size 32 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" "${EXTRA[@]}"
echo "Job finished at: $(date)"
