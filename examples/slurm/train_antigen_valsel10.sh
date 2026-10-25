#!/bin/bash
#SBATCH --job-name=valsel10
#SBATCH --partition=evo_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/valsel10_%j.out
#SBATCH --error=logs/valsel10_%j.err

# Base antiGen trained N times, each stopping at the epoch a RANDOM VALIDATION HOLDOUT
# chooses, for an arm whose final predictor is the mean of the N runs' log-probs.
#
# WHY N RUNS AND NOT ONE. A single holdout decides the epoch from 10% of an already
# small training pool -- 294 of 2,934 examples -- so the selected epoch is itself a
# noisy draw, and s47 measured antiGen's three-seed spread on the figure's panels C/D
# at 0.028-0.035, wider than the whole selection effect it was testing. Averaging the
# log-probs over N independent (split, init) draws is the cheap way to stop reporting
# one draw of that noise.
#
# EACH RUN GETS ITS OWN DIRECTORY, because `run_prefix` for this model type is
# `simple_d<d>_n<n>_lr<lr>_<timestamp>` and carries NO arm tag -- ten runs in one
# directory are ten files distinguishable only by second-of-start, and the staging
# symlinks downstream have to point at a specific seed's `_best.pt`. Mirrors the layout
# NOTES s47 used for the three-seed arm.
#
# ALL N IN ONE JOB, sequentially: a run is ~3 min and the queue is not, so ten jobs
# would be ten queue waits for 30 minutes of GPU.
#
# THE EPOCH BUDGET IS NOT THE PUBLISHED 32, and that is the point of the arm rather
# than a deviation from it: selection cannot choose an epoch outside the budget it is
# given, so a budget that binds would report a truncated search. s47's 64-epoch antiGen
# runs selected epochs 46-49 on 3,648 examples at 57 batches/epoch; this corpus holds
# 2,934, and holding out a tenth leaves 41 batches/epoch, so an epoch here is ~72% of a
# published one in optimisation steps and s47's minimum would land near 67. EPOCHS
# defaults to 96 for headroom, and the selected epochs MUST be checked to be interior
# -- any run selecting the last epoch means this budget binds too.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train}
EXAMPLES=${EXAMPLES:-$REV/examples_2021_cur.txt}
OUT_ROOT=${OUT_ROOT:-/large_storage/hielab/ispecht/fitness-inference/valsel10_aa20}
N=${N:-10}
EPOCHS=${EPOCHS:-96}
VAL_FRAC=${VAL_FRAC:-0.1}
EXTRA=("$@")
REPO_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "examples = $EXAMPLES"
echo "out-root = $OUT_ROOT"
echo "N        = $N   epochs = $EPOCHS   val-frac = $VAL_FRAC"
echo "extra    = ${EXTRA[*]}"

mkdir -p "$OUT_ROOT"
rc=0
for i in $(seq 0 $((N - 1))); do
    # --split-seed picks the holdout; --seed picks the initialisation. Both move, so
    # each run is an independent draw rather than ten readings of one split. The two are
    # offset so a log line cannot be read as the wrong one.
    SPLIT_SEED=$i
    INIT_SEED=$((100 + i))
    DIR="$OUT_ROOT/antigen_vs_s$i"
    echo ""
    echo "=== run $i/$((N - 1)): split-seed=$SPLIT_SEED init-seed=$INIT_SEED -> $DIR ==="
    # Resume rather than redo. A run is identified by its directory, and a directory
    # holding a `_best.pt` has already answered this (split, init) draw -- so a job that
    # was cancelled, or a cutoff whose runs were produced by an earlier submission, is
    # picked up where it stopped instead of spending the GPU again. Delete the directory
    # to force a rerun. Same convention as train_esm2_valsel10.sh.
    if ls "$DIR"/fitness_*_best.pt >/dev/null 2>&1; then
        echo "    already has a _best.pt; skipping (delete the directory to redo it)"
        continue
    fi
    mkdir -p "$DIR"
    antiGen-train \
        --model-type transformer \
        --cutoff 1.0 --ignore-dead-ends \
        --d-model 128 --num-layers 2 \
        --epochs "$EPOCHS" --batch-size 64 --lr 0.0001 \
        --val-mode random --val-frac "$VAL_FRAC" \
        --split-seed "$SPLIT_SEED" --seed "$INIT_SEED" --save-best \
        --examples "$EXAMPLES" \
        --data-dir "$REV" \
        --output-dir "$DIR" "${EXTRA[@]}" || { echo "RUN $i FAILED"; rc=1; }
done

echo ""
echo "=== selected epochs, and the budget check ==="
# THE CHECK, NOT A NOTE ASKING FOR ONE. This block used to print the file names and tell
# the reader to confirm each selected epoch was interior; nobody does that, and on the
# 2023-01-01 corpus NINE OF TEN runs selected the last epoch while the job exited OK.
# A run that selects the last epoch has not chosen an epoch -- its validation curve was
# still descending when the budget ran out -- so the arm collapses to "train to the
# budget", which is what the base rung already is. That is a failure, and it is loud now.
for i in $(seq 0 $((N - 1))); do
    DIR="$OUT_ROOT/antigen_vs_s$i"
    f=$(ls "$DIR"/fitness_*_best.pt 2>/dev/null | head -1)
    if [ -z "$f" ]; then echo "  s$i: NO _best.pt"; rc=1; continue; fi
    ep=$(python -I "$REPO_SELF/best_epoch.py" "$DIR")
    set -- $ep
    echo "  s$i: best epoch $1  NLL $2  (last $3, holdout $4 mutations)"
    if [ "$1" != "?" ] && [ -n "$3" ] && [ "$1" = "$3" ]; then
        echo "       ^^ SELECTED THE LAST EPOCH: this budget BINDS"
        rc=1
    fi
done

echo "Job finished at: $(date)"
[ $rc -eq 0 ] && echo "VALSEL10 OK" || { echo "VALSEL10 FAILED"; exit 1; }
