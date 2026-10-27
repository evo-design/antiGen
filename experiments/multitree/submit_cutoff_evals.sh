#!/bin/bash
# Panel M: one eval per train/test cutoff, 2020-04-01 .. 2024-01-01 (NOTES s65, s67).
#
# Six lines on the panel, from two trained rungs and four model-independent ones:
# fixed-32 base antiGen, ESM-2 35M partial FT, where x which, PSSM, BLOSUM62, EVEscape.
# One eval per cutoff with every line present, rather than the published arm's one
# directory per model -- so the figure reads rows by NAME instead of by position, which
# is what `fig2_sc2_benchmarks.R` does today and what makes its loader fragile.
#
# WHY A STAGING DIRECTORY. `discover_checkpoints` globs `fitness_*.pt` one level deep,
# and `cutoff_series/<D>/` holds `base/`, `esm2_35m_partial/`, `valsel/` (ten runs) and
# `valsel_ep32/`. Pointing the eval at the cutoff directory finds nothing; pointing it
# at a parent would find the twenty val-selected runs too, which are members of a bar
# on another panel and not rungs here. So each cutoff gets a directory of two symlinks.
#
# RE-RUNNABLE, AND THAT TOOK TWO GOES. A cutoff whose ESM-2 run has not landed is
# skipped, one already evaluated is skipped unless FORCE=1 -- and one already IN THE
# QUEUE is skipped, which the first version did not check. Running it again to pick up a
# newly-trained cutoff resubmitted every job still pending from the previous run: five
# duplicates, two of them against a directory a running job was writing. "Has it
# finished" is not the same question as "is it needed", and only the second one is
# safe to submit on. The queue is checked by job NAME, which is why every job here is
# named `cut_<D>`.
#
# NICED. The s67 scheduling lesson: this is bulk work behind the main eval's critical
# path, and sixteen jobs ahead of it in the queue is exactly what went wrong last time.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
SERIES=$STORE/cutoff_series
STAGE_ROOT=$STORE/cutoff_stage_aa20
# BLOSUM62 is one of panel M's six lines and it is a CHECKPOINT here, not a fixed map
# in data/baselines -- so the first version of this script, which staged only the two
# cutoff-specific checkpoints, produced per-cutoff CSVs with no blosum62 row at all.
# The matrix does not depend on the cutoff, so one checkpoint serves every point; what
# varies is the test window it is scored on, which is exactly what the panel plots.
BLOSUM=$STORE/ckpt_stage_aa20/fitness_blosum62_20261023_123651.pt
OUT_ROOT=${OUT_ROOT:-$STORE/cutoff_evals_aa20}
ALPHABET=ACDEFGHIKLMNPQRSTVWY
FORCE=${FORCE:-0}
# NOT evo_gpu_priority, which eval_all_sc2.sh's own #SBATCH line selects. That
# partition is for VERY SHORT jobs only (user directive, NOTES s74); a cutoff eval
# runs 4 minutes to 1h14m and does not qualify. Overridden here rather than in
# eval_all_sc2.sh, which other arms share.
# gpu_batch for a full eval. For a CPU resume use PARTITION=cpu_batch, NOT cpu:
# `cpu` is the default partition and carries QoS=cpu_interact, which caps a user at
# TWO concurrent jobs, so a fourteen-job sweep crawls two at a time with hundreds of
# cores idle. cpu_batch is the same nodes with MaxJobsPU=20 (CLAUDE.md).
PARTITION=${PARTITION:-gpu_batch}
# eval_all_sc2.sh asks for a GPU in its own header. A full cutoff eval wants one; a
# resume that scores only the factored baselines does not, and `eval.py` picks its
# device with `torch.cuda.is_available()` and falls back cleanly. GRES=NONE with
# PARTITION=cpu runs those on the idle CPU pool instead of queueing behind the GPU work.
GRES=${GRES:-gpu:1}
# eval_all_sc2.sh asks for 256G and 8 CPUs, sized for a full eval that builds ESM-2
# 650M. A resume that scores only the factored baselines never loads a network, and
# that request is what left thirteen of these PENDING on a partition with hundreds of
# idle cores: "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher
# priority partitions" is what an oversized ask looks like from the outside.
CPUS=${CPUS:-8}
MEM=${MEM:-256G}
# NICE belongs to the caller too. 1000 is right for a sixteen-job sweep behind the
# critical path and wrong for a handful of short resumes, where it simply pins the
# priority at 1 (NOTES s80).
NICE=${NICE:-1000}
# PREEMPTIBLE is the way to get a short resume onto the cluster NOW. `cpu_batch` is
# capped account-wide (QOS cpu_batch, GrpTRES=cpu=288 for all of hielab), so a sweep
# queues behind the whole lab; `cpu_preemptible` carries QOS `preempt`, which has NO
# GrpTRES and MaxJobsPU=100, and that partition runs with PreemptMode=REQUEUE.
#
# WHAT MAKES THAT SAFE HERE, and it is not general: `eval.py` resumes from its own
# output CSVs at per-MODEL granularity, so a job killed and requeued re-reads what it
# already wrote and carries on. The worst a preemption costs is the model in flight.
# --open-mode=append keeps the log across requeues instead of truncating it, which is
# the only way to tell a preemption from a crash afterwards.
# TIME, because the header's 12h is both unbackfillable for a 16-minute job and
# ILLEGAL on the short partitions: quick_preemptible has MaxTime=02:00:00 and rejects
# the submission outright with "Requested time limit is invalid". Measured runtime of
# a cutoff resume is ~16 min (jobs 3460361/3460362).
TIME=${TIME:-}
TIME_ARGS=()
[ -n "$TIME" ] && TIME_ARGS=(--time "$TIME")
REQUEUE_ARGS=()
[ "${REQUEUE:-0}" = "1" ] && REQUEUE_ARGS=(--requeue --open-mode=append)

# The decision of s67: beyond 2024-01-01 the test window falls under ~7,000 records
# (2,232 by 2025-01, 170 by 2026-01), so later points would plot as equals while
# resting on a fraction of the evidence.
[ -f "$BLOSUM" ] || { echo "no blosum62 checkpoint at $BLOSUM" >&2; exit 1; }

# The lines of panel M, by the name the eval writes. A cutoff whose CSV lacks any of
# them is re-run rather than counted as done -- which is the whole point of listing
# them here, and is what brings an older cutoff up to a line added later.
#
# `wxw_dirichlet` was added to eval.py after twelve of the fourteen cutoffs had already
# been evaluated, so they carry six rows and the two newest carry seven. Naming it here
# marks those twelve [redo]. The re-run is cheap: eval.py resumes from its own CSVs, so
# it skips the models already there AND the baseline sweep (eval.py:1002 guards the
# sweep on the baseline rows already being loaded), leaving one analytic model to score.
PANEL_ROWS="simple_d128_n2_lr0.0001 esm2_esm2_t12_35M_UR50D_lr0.0001_partial4
            gated_static_xfmr_pssm_lr0.0001
            where_times_which wxw_dirichlet pssm_root blosum62 EVEscape"

CUTOFFS="2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01
         2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01
         2023-04-01 2023-07-01 2023-10-01 2024-01-01"

cd /home/ivan.specht/antiGen

# The queue is read ONCE, up front, and a failed read aborts rather than reading as an
# empty queue. `squeue 2>/dev/null | grep -c` returns a confident 0 when the scheduler
# is merely busy, and a confident 0 here means "submit a duplicate" -- which is how
# 2021-10-01 got two jobs even after the in-flight check was added.
if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed, so the queue is unknown and submitting anything now risks" >&2
    echo "duplicating a job already in it. Nothing submitted:" >&2
    echo "$QUEUED" >&2
    exit 1
fi

n_sub=0; n_wait=0; n_done=0; n_queued=0
for D in $CUTOFFS; do
    base=$(ls "$SERIES/$D/base"/fitness_*.pt 2>/dev/null | head -1 || true)
    esm=$(ls "$SERIES/$D/esm2_35m_partial"/fitness_*.pt 2>/dev/null | head -1 || true)
    if [ -z "$base" ] || [ -z "$esm" ]; then
        echo "[wait]  $D: base=$([ -n "$base" ] && echo y || echo n) esm2=$([ -n "$esm" ] && echo y || echo n)"
        n_wait=$((n_wait + 1)); continue
    fi
    out=$OUT_ROOT/$D
    # "Done" means the CSV carries every line the panel draws, not merely that a CSV
    # exists. The first version meant the latter, and the first version also forgot to
    # stage blosum62 -- so a cutoff could be "done" while missing a line forever, and
    # re-running would never notice. Checking the rows makes the script self-healing.
    csv=$out/quantile_hit_rates_all_all-models.csv
    if [ "$FORCE" != "1" ] && [ -f "$csv" ]; then
        missing=""
        for row in $PANEL_ROWS; do
            cut -d, -f1 "$csv" | grep -qx "$row" || missing="$missing $row"
        done
        if [ -z "$missing" ]; then
            echo "[done]  $D: already evaluated"
            n_done=$((n_done + 1)); continue
        fi
        echo "[redo]  $D: CSV is missing$missing"
    fi
    # Already queued or running. Note this is checked BEFORE the staging directory is
    # touched: `rm -rf` on a stage a running job still resolves its checkpoint symlinks
    # through is the other half of the same mistake.
    inflight=$(printf '%s\n' "$QUEUED" | grep -cx "cut_$D" || true)
    if [ "$inflight" != "0" ]; then
        echo "[queue] $D: $inflight job(s) already in the queue"
        n_queued=$((n_queued + 1)); continue
    fi
    stage=$STAGE_ROOT/$D
    rm -rf "$stage"; mkdir -p "$stage" "$out"
    ln -s "$base" "$stage/$(basename "$base")"
    ln -s "$esm"  "$stage/$(basename "$esm")"
    ln -s "$BLOSUM" "$stage/$(basename "$BLOSUM")"
    # The antiGen/PSSM gate, if this cutoff has one trained yet. Staged WITHOUT being
    # named in PANEL_ROWS on purpose: an eval that runs picks the gate up and writes
    # its row, but a cutoff is not marked [redo] for lacking it.
    #
    # NAMING IT IN PANEL_ROWS BEFORE EVERY GATE IS TRAINED WOULD LOOP FOREVER. The
    # redo check asks whether the CSV carries each listed row; a cutoff whose gate
    # checkpoint does not exist can never produce that row, so the reconciler would
    # resubmit it every cycle and never converge. Add the row to PANEL_ROWS only once
    # all sixteen checkpoints are on disk -- that is a one-line change and it then
    # correctly marks all sixteen [redo], which is what putting a new line on the
    # panel costs.
    gate=$(ls "$SERIES/$D/gate_xfmr_pssm"/fitness_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$gate" ]; then
        ln -s "$gate" "$stage/$(basename "$gate")"
    else
        echo "        (no gate checkpoint yet for $D; staged without it)"
    fi

    # The corpus is per cutoff; the TEST corpus is the one shared file, cut at the same
    # date by --split-date. Reading it any other way would give each point a different
    # test set AND a different split rule (NOTES s65).
    CKPT_DIR=$stage OUTDIR=$out DATA_DIR="" \
    sbatch --job-name="cut_$D" --nice="$NICE" --partition="$PARTITION" --gres="$GRES" \
        --cpus-per-task="$CPUS" --mem="$MEM" \
        "${TIME_ARGS[@]}" "${REQUEUE_ARGS[@]}" \
        examples/slurm/eval_all_sc2.sh \
        --alphabet $ALPHABET \
        --train-examples "$REVISED/sc2-$D/train/examples_${D}_cur.txt" \
        --test-examples  "$REVISED/sc2-test/examples_all_cur.txt" \
        --split-date "$D" \
        --root-counts    "$REVISED/sc2-$D/train/counts_${D}_cur.tsv" \
        --variant-counts-dir "$REVISED/sc2-test/counts_ba2" \
        --counts-since "$D" \
        --multitree-baselines
    n_sub=$((n_sub + 1))
done
echo ""
echo "submitted $n_sub, already done $n_done, already queued $n_queued, waiting on a training $n_wait"
