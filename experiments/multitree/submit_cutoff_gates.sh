#!/bin/bash
# The antiGen / PSSM gate across the panel-M cutoff sweep (NOTES s82).
#
# Sixteen trainings, one per cutoff, each the base recipe plus the gate. Re-runnable:
# a cutoff already trained is skipped, and one already in the queue is skipped by job
# NAME -- the s70 rule, because "has it finished" is not "is it needed".
set -euo pipefail

STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
SERIES=$STORE/cutoff_series
REVISED=/home/ivan.specht/antiGen-data-revised/output
PARTITION=${PARTITION:-gpu_batch}
NICE=${NICE:-0}
# TIME, and the default is short because the job is short. The gate is the BASE recipe
# plus a gate module -- 32 epochs of simple_d128_n2 -- and the base training at
# 2022-01-01 took 10m25s (cutoff_series log 3428706: 14:31:20 -> 14:41:45). The 6-11h
# figure quoted earlier in this work was the whole `cutoff_series` JOB, which is base
# plus TEN 96-epoch val-selected runs: 992 epochs against 32. A 14h limit on a
# ten-minute job is unbackfillable, which is why the first pilot never started while
# seven GPUs sat free.
TIME=${TIME:-01:00:00}

CUTOFFS=${CUTOFFS:-"2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01
         2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01
         2023-04-01 2023-07-01 2023-10-01 2024-01-01"}

cd "$REPO"
if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed, so the queue is unknown; nothing submitted" >&2
    exit 1
fi

n_sub=0; n_done=0; n_queued=0; n_wait=0
for D in $CUTOFFS; do
    [ -f "$REVISED/sc2-$D/train/examples_${D}_cur.txt" ] || {
        echo "[wait]  $D: no training corpus"; n_wait=$((n_wait + 1)); continue; }
    if ls "$SERIES/$D/gate_xfmr_pssm"/fitness_*.pt >/dev/null 2>&1; then
        echo "[done]  $D: gate already trained"; n_done=$((n_done + 1)); continue
    fi
    if printf '%s\n' "$QUEUED" | grep -qx "gate_$D"; then
        echo "[queue] $D: already in the queue"; n_queued=$((n_queued + 1)); continue
    fi
    sbatch --job-name="gate_$D" --partition="$PARTITION" --nice="$NICE" \
        --time "$TIME" \
        examples/slurm/train_cutoff_gate.sh "$D" >/dev/null
    echo "[sub]   $D"
    n_sub=$((n_sub + 1))
done
echo ""
echo "submitted $n_sub, already trained $n_done, already queued $n_queued, no corpus $n_wait"
