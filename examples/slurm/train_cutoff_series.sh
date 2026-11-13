#!/bin/bash
#SBATCH --job-name=cutoff_series
#SBATCH --partition=gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/cutoff_series_%j.out
#SBATCH --error=logs/cutoff_series_%j.err

# ONE CUTOFF of the train/test-date sweep: base antiGen plus the ten val-selected runs
# whose mean of log-probs is the s56 bar. Twenty-four of these, one per quarter that
# antiGen-data-revised regenerated, rebuild the substrate for an analogue of
# `sc2_results` panel M -- recall at the top decile as a function of where the
# train/test boundary is drawn.
#
# WHAT PANEL M NEEDS, and why this is only half of it. That panel reads one
# `quantile_hit_rates_all_*.csv` per (model, cutoff) and plots recall against the
# cutoff. This job produces the CHECKPOINTS for one cutoff; the eval that turns them
# into that CSV is a separate pass, because it needs the test corpus read at this
# cutoff's split date and that is one `--split-date` away from the figure arm's eval.
#
# TWO PREDICTORS PER CUTOFF, deliberately:
#   base      the published recipe, 32 epochs chosen on nothing, which is the rung
#             every published figure draws.
#   valsel    ten runs, each stopping at the epoch its own random 10% holdout chose,
#             averaged as the mean of their log-probs (NOTES s56). The holdout is well
#             powered at every cutoff but the smallest -- 2021's was 854 mutations and
#             2023-01-01's is 23,620 -- so what limits this arm is the BUDGET, not the
#             evidence.
#
# THE BUDGET IS 96 EPOCHS, NOT THE PUBLISHED 32, and that is a correction rather than a
# preference. A first pass at this sweep ran the val-selected half at 32 and NINE OF TEN
# runs at 2023-01-01 selected epoch 31 -- the last one -- with the same thing true to
# some degree at every cutoff (9/10 at 2020-04-01, 7/10 at 2021-01-01, 5/8 at
# 2024-07-01). A run that selects its last epoch has not chosen an epoch: its held-out
# curve was still descending when the budget ran out, so the arm collapses to "train to
# the budget", which is exactly what `base` already is. Those runs are kept as
# `valsel_ep32/` -- a ten-seed average at the published budget is the control s56's Open
# asked for -- and the arm is re-run at 96. `train_antigen_valsel10.sh` now FAILS a job
# whose runs select the last epoch, so a budget that still binds says so instead of
# exiting OK.
#
# `base` stays at 32, because that is the published recipe and the rung the figures draw.
# Only the selection arm needs room to find a minimum.
#
# THE RECIPE DOES NOT MOVE ACROSS CUTOFFS, which is the whole point of a sweep: 32
# epochs, batch 64, lr 1e-4, `simple_d128_n2`, split seed i and init seed 100+i. So an
# epoch at 2026-01-01 is ~400x the optimisation steps of one at 2020-04-01 (75,647
# records against 190), and the curve panel M draws is a curve in corpus size as much
# as in date. That is a property of the question, not of this script, and it is worth
# stating wherever the panel is read.
#
# RESUME, DO NOT REDO. Both halves skip work already on disk, so a cancelled job or a
# cutoff whose runs came from an earlier submission costs nothing to re-submit.
#
# USAGE: sbatch examples/slurm/train_cutoff_series.sh 2022-07-01

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

D=${1:?usage: train_cutoff_series.sh YYYY-MM-DD}
REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/cutoff_series}
N=${N:-10}
EPOCHS=${EPOCHS:-96}

REV="$DATA/sc2-$D/train"
EXAMPLES="$REV/examples_${D}_cur.txt"
ROOT="$SERIES/$D"

echo "cutoff   = $D"
echo "examples = $EXAMPLES"
echo "root     = $ROOT"
[ -d "$REV" ]      || { echo "ABORT: no train dir $REV"; exit 1; }
[ -f "$EXAMPLES" ] || { echo "ABORT: no examples at $EXAMPLES"; exit 1; }
mkdir -p "$ROOT/base" "$ROOT/valsel"
rc=0

# A per-cutoff manifest, so the directory says what it is without the job log. Written
# first: a job that dies half way still leaves the reader something to go on.
cat > "$ROOT/MANIFEST.txt" <<EOF
cutoff        $D
examples      $EXAMPLES
records       $(grep -c '^EXAMPLE_ID:' "$EXAMPLES")
counts        $REV/counts_${D}_cur.tsv
recipe        simple_d128_n2, 32 epochs, batch 64, lr 1e-4, 20-letter alphabet
base          base/           published recipe, no validation, no epoch selection
valsel        valsel/antigen_vs_s<i>   --val-mode random --val-frac 0.1 --save-best,
              split seed i, init seed 100+i, i = 0..$((N - 1)); the bar is the mean of
              the ten runs' log-probs (NOTES s56)
test corpus   $DATA/sc2-test/examples_all_cur.txt read at --split-date $D
submitted     $(date -Iseconds) by $USER, job $SLURM_JOB_ID
EOF
echo ""; echo "=== manifest ==="; cat "$ROOT/MANIFEST.txt"

echo ""
echo "=== 1. base antiGen ==="
if ls "$ROOT/base"/fitness_*.pt >/dev/null 2>&1; then
    echo "  already trained; skipping (delete $ROOT/base to redo it)"
else
    DATA_DIR="$REV" OUT_DIR="$ROOT/base" \
        bash "$REPO/examples/slurm/train_transformer.sh" --examples "$EXAMPLES" \
        || { echo "BASE FAILED"; rc=1; }
fi

echo ""
echo "=== 2. the $N val-selected runs ==="
REV="$REV" EXAMPLES="$EXAMPLES" OUT_ROOT="$ROOT/valsel" N="$N" EPOCHS="$EPOCHS" \
    bash "$REPO/examples/slurm/train_antigen_valsel10.sh" \
    || { echo "VALSEL FAILED"; rc=1; }

echo ""
echo "=== what this cutoff now holds ==="
b=$(ls "$ROOT/base"/fitness_*.pt 2>/dev/null | wc -l)
v=$(ls -d "$ROOT/valsel"/antigen_vs_s*/ 2>/dev/null | while read d; do
        ls "$d"/fitness_*_best.pt >/dev/null 2>&1 && echo x; done | wc -l)
echo "  base checkpoints: $b"
echo "  val-selected runs with a _best.pt: $v of $N"
[ "$b" -ge 1 ] && [ "$v" -eq "$N" ] || { echo "INCOMPLETE"; rc=1; }

echo "Job finished at: $(date)"
[ $rc -eq 0 ] && echo "CUTOFF_SERIES $D OK" || { echo "CUTOFF_SERIES $D FAILED"; exit 1; }
