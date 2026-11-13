#!/bin/bash
#SBATCH --job-name=valsel_seed
#SBATCH --partition=gpu_batch,gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/valsel_seed_%A_%a.out
#SBATCH --error=logs/valsel_seed_%A_%a.err

# ONE val-selected run of the cutoff series, as its own job.
#
# `train_cutoff_series.sh` runs all ten in series, which is right for the small 2021
# corpus. At a late cutoff it is not: sc2-2023-10-01 has 64,715 training examples
# against 2021's 2,934, so an epoch is ~1,011 steps rather than ~46 and ten runs at 96
# epochs is roughly twelve hours in one job -- too long to backfill and serial for no
# reason. As an array of ten one-seed jobs each is about an hour and they run at once.
#
# THE RECIPE IS UNCHANGED. This calls `train_antigen_valsel10.sh` with SEEDS set to a
# single seed; split seed i and init seed 100+i are still derived inside it, so a run
# produced here is bit-identical to the same seed produced by the series script.
#
# RESUME: a seed whose directory already holds a _best.pt is skipped.
#
# USAGE: sbatch --array=0-9 reproduce/slurm/train_valsel_seed.sh 2023-10-01
set -uo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID  array task: ${SLURM_ARRAY_TASK_ID:-<none>}"

D=${1:?usage: train_valsel_seed.sh YYYY-MM-DD}
REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/cutoff_series}
N=${N:-10}
EPOCHS=${EPOCHS:-96}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}

REV="$DATA/sc2-$D/train"
EXAMPLES="$REV/examples_${D}_cur.txt"
OUT_ROOT="$SERIES/$D/valsel"

echo "cutoff   = $D"
echo "seed     = $SEED of 0..$((N - 1))"
echo "examples = $EXAMPLES"
[ -f "$EXAMPLES" ] || { echo "ABORT: no examples at $EXAMPLES"; exit 1; }
mkdir -p "$OUT_ROOT"

REV="$REV" EXAMPLES="$EXAMPLES" OUT_ROOT="$OUT_ROOT" N="$N" EPOCHS="$EPOCHS" \
    SEEDS="$SEED" bash "$REPO/reproduce/slurm/train_antigen_valsel10.sh"
rc=$?

echo "Job finished at: $(date)"
[ $rc -eq 0 ] && echo "VALSEL SEED $SEED OK" || { echo "VALSEL SEED $SEED FAILED"; exit 1; }
