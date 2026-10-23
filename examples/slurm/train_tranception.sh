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

# DATA_DIR / OUT_DIR / MSA / METADATA are env overrides, and any extra arguments are
# passed straight through to antiGen-train, so a retrain on a rebuilt corpus reuses this
# exact recipe instead of a fork that would drift. The defaults reproduce the published
# checkpoints.
#
# MSA AND METADATA ARE PARAMETERISED, which they were not before s55, and the reason is
# that the claim they were not tree-dependent has stopped being true. The published
# inputs are the open-data spike MSA plus its metadata, filtered to --date-cutoff at
# load time. antiGen-data-revised writes its OWN MSA per run, built from the same
# sequences it generated the examples from and so already restricted to the training
# window -- and with no metadata TSV beside it, because the headers carry no dates.
#
# `METADATA=` empty drops --metadata, which means "use the MSA as given". That is a
# claim about the file and cannot be checked: a FASTA carries no dates, so an
# unfiltered PANDEMIC MSA would pretrain on the test window and nothing downstream
# could tell. Only ever pair it with an MSA that is window-restricted by construction.
DATA_DIR=${DATA_DIR:-/home/ivan.specht/antiGen-data/output1}
OUT_DIR=${OUT_DIR:-/large_storage/hielab/ispecht/fitness-inference}
MSA=${MSA:-/home/ivan.specht/sc2-open-data/spike.fasta}
METADATA=${METADATA-/home/ivan.specht/sc2-open-data/metadata.tsv}
META_ARG=()
[ -n "$METADATA" ] && META_ARG=(--metadata "$METADATA")
EXTRA=("$@")
echo "data-dir = $DATA_DIR"
echo "out-dir  = $OUT_DIR"
echo "msa      = $MSA"
echo "metadata = ${METADATA:-<none: the MSA is taken as already window-restricted>}"
echo "extra    = ${EXTRA[*]}"

antiGen-train \
    --model-type tranception \
    --msa "$MSA" "${META_ARG[@]}" \
    --cutoff 1.0 --ignore-dead-ends \
    --lr 0.0001 --epochs 32 \
    --batch-size 32 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" "${EXTRA[@]}"
echo "Job finished at: $(date)"
