#!/bin/bash
# The backbone-only test arm of one non-ncov pathogen: h3n2, rsv-a or denv2, one
# row of slurm/arms.tsv each -- the same table run_arms.sbatch reads, so the tree
# and the cutoff are by construction the ones the placed test arm used.
#
# This is run_arms.sbatch's test side with the attachment step taken out. No
# `prepare` and no `place`: just `sets` on the tree that is already in the
# pathogen's folder, which reads the Nextstrain tree, puts Hartigan allele sets
# at every (node, site), and resolves/collapses/splits that into examples. The
# queue is never touched, so there is no counts matrix, no clock and no MSA --
# nothing here is a function of the open-data sequences.
#
# --post-cutoff D still means what it means on the placed arm: a record is at or
# after D when *every observed date below it* is, and the observed dates here are
# the tree's own tips' num_date rather than the dates of placed sequences. The
# tips are a subset of what the placed arm saw, so a node's earliest date below
# can only be later here, and the True side can only grow. The two test sets are
# therefore not nested -- they are two different trees, read the same way.
#
# Seconds, not minutes: the whole cost of the placed arm was the placement.
#
#SBATCH --job-name=antigen-arms-bb
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=work/arms/run_bb_%A_%a.log
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
SEED=${ANTIGEN_SEED:-1}
DATA=${ANTIGEN_DATA:-$HOME}
WORK=work/arms/$ARM
OUT=output/$ARM-test-backbone
mkdir -p "$WORK" "$OUT"

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_TASK_ID:-none})"
echo "arm    : $ARM"
echo "cutoff : $D"
echo "gene   : $GENE"
echo "test   : $TEST_TREE / $TEST_ROOT"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "seed   : $SEED"
echo "start  : $(date)"

echo
echo "=== backbone test: the $(basename "$TEST_TREE") tree alone, flagged at $D ==="
echo

"$BIN" sets \
    --gene "$GENE" \
    --tree "$DATA/$TEST_TREE" \
    --root-seq "$DATA/$TEST_ROOT" \
    --seed "$SEED" \
    --post-cutoff "$D" \
    --out-tree "$WORK/tree_backbone.tsv" \
    --out "$WORK/sets_backbone.tsv" \
    --out-examples "$WORK/examples_backbone.txt"

echo
echo "built  : $(date)"
echo

cp "$WORK/examples_backbone.txt" "$OUT/examples_all_cur.txt"
cp "$WORK/tree_backbone.tsv"     "$OUT/tree_all_cur.tsv"
cp "$WORK/sets_backbone.tsv"     "$OUT/sets_all_cur.tsv"
echo "copied : $OUT"
echo

set +e
python3 -I check_examples.py "$WORK/examples_backbone.txt"
echo "checker: exit $?"
set -e

echo
echo "done   : $(date)"
