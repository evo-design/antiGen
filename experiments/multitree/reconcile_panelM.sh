#!/bin/bash
# Drive panel M to its desired state, repeatedly and without supervision.
#
# WHY THIS EXISTS. Every job this drives is already idempotent -- the two submit
# scripts check what is on disk and what is in the queue before submitting anything --
# but nothing was CALLING them again when a job died. A TIMEOUT at 02:00:16 or a
# preemption then sat unnoticed until a human looked, which across one night cost more
# time than every job combined.
#
# This is a reconciler, not an event handler: it asks what is MISSING rather than what
# FAILED, so it cannot be confused by the shape of a failure it did not anticipate --
# a timeout, a preemption, a node dying, a submission rejected for quota, or a job that
# exited 0 having done nothing. If the artifact is absent and nothing is queued for it,
# it goes back in.
#
# RESOURCES ARE THE MEASURED ONES, not guesses:
#   gates        ~30 s each. quick_preemptible (PriorityTier=10) starts them in
#                seconds; preemptible (tier 1) and cpu_preemptible (tier 0) never did.
#   cutoff evals most are ~16 min on 4 CPUs. 2022-07-01 and 2022-10-01 are NOT: they
#                timed out at 45 min on 4 CPUs and again at 2 h on 16, while the same
#                work took 56 min on a GPU. So those two get a GPU and the rest do not.
#
# PARTITION TIER IS THE THING THAT PREDICTS WHETHER ANYTHING RUNS, and it is not in
# squeue's reason string. tier 1000 = gpu_batch/cpu_batch (capped), tier 10 =
# quick_preemptible (MaxJobsPU=4, MaxSubmitPU=8, MaxTime=2h), tier 1 = preemptible,
# tier 0 = cpu_preemptible. A quota rejection here is not an error: the next cycle
# retries once a slot frees.
set -uo pipefail

REPO=/home/ivan.specht/antiGen
STORE=/large_storage/hielab/ispecht/fitness-inference
OUT=$STORE/cutoff_evals_aa20
SERIES=$STORE/cutoff_series
cd "$REPO"

CUTOFFS="2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01
         2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01
         2023-04-01 2023-07-01 2023-10-01 2024-01-01"
# EVERY cutoff now needs a GPU, not just the two slow ones. The row that is missing
# is no longer `wxw_dirichlet`, an analytic map, but the GATE -- a 2-layer transformer
# forward pass over every test query. The analytic version already timed out on CPU at
# the two largest cutoffs; a network pass is strictly heavier, so routing the sweep to
# CPU would be discovering that again sixteen times.
SLOW=${SLOW:-"2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01
         2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01
         2023-04-01 2023-07-01 2023-10-01 2024-01-01"}

missing_evals=""; missing_gates=""
for D in $CUTOFFS; do
    c=$OUT/$D/quantile_hit_rates_all_all-models.csv
    # THE ROWS THIS CHECKS MUST TRACK `PANEL_ROWS`. It originally looked for
    # `wxw_dirichlet` alone; once the gate was named in PANEL_ROWS that test would have
    # reported RECONCILED while every cutoff still lacked the gate row, and the sweep
    # would have stopped one line short of the panel it is driving. Listed explicitly
    # rather than parsed out of the other script, because a silent disagreement between
    # the two is exactly the failure this reconciler exists to prevent.
    want_rows="wxw_dirichlet gated_static_xfmr_pssm_lr0.0001"
    ok=1
    if [ -f "$c" ]; then
        for r in $want_rows; do
            cut -d, -f1 "$c" | grep -qx "$r" || ok=0
        done
    else
        ok=0
    fi
    [ "$ok" = "1" ] || missing_evals="$missing_evals $D"
    ls "$SERIES/$D/gate_xfmr_pssm"/fitness_*.pt >/dev/null 2>&1 || \
        missing_gates="$missing_gates $D"
done

n_e=$(echo $missing_evals | wc -w); n_g=$(echo $missing_gates | wc -w)
echo "[$(date +%H:%M)] evals missing: $n_e of 16 | gates missing: $n_g of 16"
[ "$n_e" = "0" ] && [ "$n_g" = "0" ] && { echo "[$(date +%H:%M)] RECONCILED: panel M has everything"; exit 0; }

# Gates first: they are seconds long, so they clear the shared MaxSubmitPU=8 budget
# quickly rather than sitting in it.
if [ "$n_g" != "0" ]; then
    PARTITION=quick_preemptible NICE=0 TIME=00:30:00 CUTOFFS="$missing_gates" \
        ./experiments/multitree/submit_cutoff_gates.sh 2>&1 | grep -E "^\[sub\]|error" || true
fi

# NORMALISED TO ONE LINE before the membership test. `SLOW` is written across several
# lines for readability, and ` $SLOW ` then keeps those newlines -- so a date sitting at
# a line boundary does not match " $D " and silently falls into the CPU branch, which is
# the opposite of what listing it was for. `$(echo $SLOW)` unquoted collapses the
# whitespace.
slow_flat=" $(echo $SLOW) "
fast=""; slow=""
for D in $missing_evals; do
    case "$slow_flat" in
        *" $D "*) slow="$slow $D" ;;
        *)        fast="$fast $D" ;;
    esac
done
if [ -n "$fast" ]; then
    PARTITION=quick_preemptible GRES=NONE CPUS=4 MEM=32G NICE=0 REQUEUE=1 TIME=00:45:00 \
        CUTOFFS="$fast" ./experiments/multitree/submit_cutoff_evals.sh 2>&1 \
        | grep -E "Submitted|error" || true
fi
if [ -n "$slow" ]; then
    out=$(PARTITION=quick_preemptible GRES=gpu:1 CPUS=8 MEM=96G NICE=0 REQUEUE=1 \
        TIME=01:50:00 CUTOFFS="$slow" \
        ./experiments/multitree/submit_cutoff_evals.sh 2>&1)
    echo "$out" | grep -E "^Submitted" || true
    # EXPECTED BACKPRESSURE, NOT A FAILURE. quick_preemptible allows MaxSubmitPU=8, so
    # a sweep larger than that has its tail rejected every cycle until slots free. Left
    # as a raw sbatch error it fires once per cycle and reads like something is broken,
    # which is how a real error would get lost in it. Counted and reported as what it
    # is; anything else from sbatch still surfaces verbatim.
    n_throttled=$(echo "$out" | grep -c "QOSMaxSubmitJobPerUserLimit" || true)
    [ "$n_throttled" != "0" ] && \
        echo "[throttle] $n_throttled submission(s) deferred to the next cycle (MaxSubmitPU=8)"
    echo "$out" | grep -E "error" | grep -vE "QOSMaxSubmitJobPerUserLimit|Job violates accounting" || true
fi
