#!/bin/bash
# The two panel-M cutoffs whose ESM-2 35M partial-FT training hit the wall clock:
# 2022-07-01 (job 3432520) and 2022-10-01 (job 3432521). Both reached epoch 8 of 32
# and were killed TIMEOUT, so `cutoff_series/<D>/esm2_35m_partial/` holds a manifest
# and no checkpoint, and `submit_cutoff_evals.sh` reports them as [wait] forever.
#
# IT IS NOT THE CORPUS SIZE. 2022-07-01 is 44,481 examples and managed 8 epochs in 11
# hours; 2022-10-01 is comparable. 2023-01-01 is 53,916 examples -- MORE data, 1,686
# batches an epoch against 1,391 -- and finished all 32 in 6h41m, and 2024-01-01 took
# 8h28m. So the two that died were running roughly 7x slower per epoch than a larger
# job on the same recipe. Whatever throttled them, it was not the work.
#
# WHAT THE TWO HAVE IN COMMON is the node: both ran on GPUCACE and nothing else in the
# sweep did, and they overlapped there for ten hours (21:27-08:27 and 21:58-09:58).
# That is consistent with two readings -- a degraded GPU, or two 256G jobs contending
# on one host -- and this retry does not need to tell them apart, so it covers both:
# GPUCACE is excluded, and the limit is 16h, about double what the largest healthy
# cutoff actually needed. Over-asking for wall clock costs queue priority; under-asking
# costs another eleven hours and a second empty directory.
#
# No resume: antiGen-train writes its checkpoint at the end, so there is nothing on
# disk for these two to pick up and both restart from scratch.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
CUTOFFS=${CUTOFFS:-"2022-07-01 2022-10-01"}

cd /home/ivan.specht/antiGen

# Same once-up-front queue read as submit_cutoff_evals.sh, for the same reason: a
# `squeue` that merely fails must not read as an empty queue and license a duplicate.
if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed; the queue is unknown and nothing is submitted:" >&2
    echo "$QUEUED" >&2; exit 1
fi

for D in $CUTOFFS; do
    out=$STORE/cutoff_series/$D/esm2_35m_partial
    if ls "$out"/fitness_*.pt >/dev/null 2>&1; then
        echo "[done]  $D: checkpoint already on disk"; continue
    fi
    if grep -qx "esm2_$D" <<<"$QUEUED"; then
        echo "[queued] $D: already in the queue"; continue
    fi
    ex="$REVISED/sc2-$D/train/examples_${D}_cur.txt"
    [ -f "$ex" ] || { echo "[skip]  $D: no examples at $ex" >&2; continue; }
    DATA_DIR="$REVISED/sc2-$D/train" OUT_DIR="$out" \
    sbatch --job-name="esm2_$D" --time=16:00:00 --exclude=GPUCACE \
        examples/slurm/train_esm2_partial.sh --examples "$ex"
done
