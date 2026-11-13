#!/bin/bash
# The train and the test arm of one non-ncov pathogen, both in one task: h3n2,
# rsv-a or denv2, one row of slurm/arms.tsv each. The sc2 shape, with one cutoff
# per pathogen instead of twenty-four -- see NOTES, "Directive, 2026-10-28
# (second): train and test arms for the three new pathogens".
#
#   train  the historical tree fetched by fetch_snapshots.py, queue filtered to
#          collection dates strictly before D, no --post-cutoff at all: both the
#          tree and the queue stop before D, so the flag would be vacuous
#   test   the tree already in the folder, the *whole* queue, --post-cutoff D.
#          The test set at D is the records whose POST_CUTOFF carries D, which is
#          exactly how the one sc2 test tree is read
#
# No --variant on either side: the directive asks for no variants or variant
# roots here, so there is no --out-variant-counts and no counts_<cutoff>/ either.
#
# The train side writes --out-msa, as slurm/run_2021.sbatch does: the alignment
# of the sequences in the training window, gaps and stops included, written as
# the queue is consumed so that it is one-to-one with counts_train.tsv beside it.
# run_series.sbatch dropped it for sc2 only because 24 arms of ~11 GB would not
# fit; here the largest is h3n2's 53 MB. The test side does not write one -- same
# as run_all.sbatch -- but it is one flag away if it is ever wanted.
#
# These are small -- the largest queue is h3n2's 202k against sc2's 9.4M -- so
# both sides of an arm fit in one task: 3 min 22 s for all three on four cores,
# of which h3n2's test placement is 174 s and everything else together is 28 s. It is a batch script for the same reason the others are, and because the
# arms table is the record of what was run; `bash slurm/run_arms.sbatch 0` runs
# one arm by hand, with no scheduler involved.
#
#SBATCH --job-name=antigen-arms
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=work/arms/run_%A_%a.log
#
# --exclude: the nodes whose /media/8TBNVME/scratch is not set up -- see NOTES,
# "It is the idle nodes, and that is schedulable around". Drop it once the
# cluster is fixed.
#SBATCH --exclude=GPU0F98,GPU115A,GPU1034,GPU104C

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

LIST=${ANTIGEN_ARMS:-slurm/arms.tsv}
IDX=${SLURM_ARRAY_TASK_ID:-${1:?need an array index or an argument}}
# Row IDX of the table, past the one header line.
IFS=$'\t' read -r ARM D GENE IMPUTE TRAIN_TREE TRAIN_ROOT TEST_TREE TEST_ROOT FASTA META \
    < <(grep -v '^#' "$LIST" | sed -n "$((IDX + 1))p")
[ -n "${ARM:-}" ] || { echo "no row $IDX in $LIST" >&2; exit 1; }

BIN=${ANTIGEN_BIN:-work/antigen_arms}
THREADS=${ANTIGEN_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
SEED=${ANTIGEN_SEED:-1}
# The table holds paths relative to the home directory, which is where the three
# pathogen folders live; nothing in it is relative to this repo.
DATA=${ANTIGEN_DATA:-$HOME}
WORK=work/arms/$ARM
mkdir -p "$WORK" "output/$ARM-$D/train" "output/$ARM-test"

# --date-impute is off unless the row asks for it. Only denv2 does: 2,823 of its
# rows carry a year and a month and no day, and a day imputed inside the month
# cannot move a record across the cutoff or across a clock month.
IMPUTE_ARG=()
[ "$IMPUTE" = "none" ] || IMPUTE_ARG=(--date-impute "$IMPUTE")

FASTA_ABS=$(echo "$FASTA" | tr ',' '\n' | sed "s|^|$DATA/|" | paste -sd,)

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_TASK_ID:-none})"
echo "arm    : $ARM"
echo "cutoff : $D"
echo "gene   : $GENE"
echo "impute : $IMPUTE"
echo "train  : $TRAIN_TREE / $TRAIN_ROOT"
echo "test   : $TEST_TREE / $TEST_ROOT"
echo "fasta  : $FASTA"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "seed   : $SEED"
echo "threads: $THREADS (the tree written does not depend on this)"
echo "start  : $(date)"

# ---------------------------------------------------------------- train ------
echo
echo "=== train: the $(basename "$TRAIN_TREE") tree, dates strictly before $D ==="
echo

"$BIN" prepare \
    --gene "$GENE" \
    --root-seq "$DATA/$TRAIN_ROOT" \
    --fasta "$FASTA_ABS" \
    --metadata "$DATA/$META" \
    --date-cutoff "$D" \
    "${IMPUTE_ARG[@]}" \
    --out "$WORK/prepared_train.tsv" \
    --out-clock "$WORK/clock_train.tsv"

echo

"$BIN" place \
    --gene "$GENE" \
    --tree "$DATA/$TRAIN_TREE" \
    --root-seq "$DATA/$TRAIN_ROOT" \
    --prepared "$WORK/prepared_train.tsv" \
    --seed "$SEED" \
    --threads "$THREADS" \
    --out-tree "$WORK/tree_train.tsv" \
    --out "$WORK/sets_train.tsv" \
    --out-examples "$WORK/examples_train.txt" \
    --out-matches "$WORK/matches_train.tsv" \
    --out-counts "$WORK/counts_train.tsv" \
    --out-msa "$WORK/msa_train.fasta"

# ----------------------------------------------------------------- test ------
echo
echo "=== test: the $(basename "$TEST_TREE") tree, every dated record, flagged at $D ==="
echo

"$BIN" prepare \
    --gene "$GENE" \
    --root-seq "$DATA/$TEST_ROOT" \
    --fasta "$FASTA_ABS" \
    --metadata "$DATA/$META" \
    --date-cutoff none \
    "${IMPUTE_ARG[@]}" \
    --out "$WORK/prepared_test.tsv" \
    --out-clock "$WORK/clock_test.tsv"

echo

"$BIN" place \
    --gene "$GENE" \
    --tree "$DATA/$TEST_TREE" \
    --root-seq "$DATA/$TEST_ROOT" \
    --prepared "$WORK/prepared_test.tsv" \
    --seed "$SEED" \
    --threads "$THREADS" \
    --post-cutoff "$D" \
    --out-tree "$WORK/tree_test.tsv" \
    --out "$WORK/sets_test.tsv" \
    --out-examples "$WORK/examples_test.txt" \
    --out-matches "$WORK/matches_test.tsv" \
    --out-counts "$WORK/counts_test.tsv"

echo
echo "placed : $(date)"
echo

rm -f "$WORK/prepared_train.tsv" "$WORK/prepared_test.tsv"

for s in train test; do
    case $s in
        train) OUT=output/$ARM-$D/train; TAG=${D} ;;
        test)  OUT=output/$ARM-test;     TAG=all  ;;
    esac
    cp "$WORK/examples_$s.txt" "$OUT/examples_${TAG}_cur.txt"
    cp "$WORK/tree_$s.tsv"     "$OUT/tree_${TAG}_cur.tsv"
    cp "$WORK/sets_$s.tsv"     "$OUT/sets_${TAG}_cur.tsv"
    cp "$WORK/counts_$s.tsv"   "$OUT/counts_${TAG}_cur.tsv"
    cp "$WORK/clock_$s.tsv"    "$OUT/clock_${TAG}_cur.tsv"
    # An `if`, not `a && b`: under `set -e` a false test as the last command of
    # an AND-list aborts the script, and only the train side writes an MSA.
    if [ -f "$WORK/msa_$s.fasta" ]; then
        cp "$WORK/msa_$s.fasta" "$OUT/msa_${TAG}_cur.fasta"
    fi
    echo "copied : $OUT"
done
echo

set +e
python3 -I check_examples.py "$WORK/examples_train.txt"
echo "checker-train: exit $?"
python3 -I check_examples.py "$WORK/examples_test.txt"
echo "checker-test: exit $?"
set -e

echo
echo "done   : $(date)"
