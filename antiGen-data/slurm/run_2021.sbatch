#!/bin/bash
# The 2021-01-01 cutoff arm, end to end: prepare the queue, then place it on the
# 2020-12-29 tree and write the examples.
#
# prepare runs here rather than being taken from disk because the queue on disk
# predates the clock envelope (Directive, 2026-10-22 (seventh)) -- it was selected
# by the old regression line, which that directive replaced for *both* runs.
#
# No --post-cutoff, and its default is no longer 2021-01-01 but no cutoff at all,
# so the examples carry an empty POST_CUTOFF. That is the intended train-side
# output: every record here is before 2021-01-01 anyway -- the tree and the queue
# both stop there -- so the flag only ever read False. Which cutoffs a record
# falls after is now asked once, on the test tree slurm/run_all.sbatch builds.
#SBATCH --job-name=antigen-2021
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=work/run_2021.%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BIN=${ANTIGEN_BIN:-work/antigen_2021}
THREADS=${ANTIGEN_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
SEED=${ANTIGEN_SEED:-1}
TREE=${ANTIGEN_TREE:-data/ncov_global@2020-12-29.json}
ROOT=${ANTIGEN_ROOT:-data/ncov_global@2020-12-29_root-sequence.json}
FASTA=${ANTIGEN_FASTA:-$HOME/sc2-open-data/spike.fasta}
META=${ANTIGEN_META:-$HOME/sc2-open-data/metadata.tsv}
OUTDIR=output/sc2-2021/train

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none}"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "seed   : $SEED"
echo "threads: $THREADS (the tree written does not depend on this)"
echo "start  : $(date)"
echo

"$BIN" prepare \
    --root-seq "$ROOT" \
    --fasta "$FASTA" \
    --metadata "$META" \
    --date-cutoff 2021-01-01 \
    --out work/prepared_2021.tsv \
    --out-clock work/clock_2021.tsv

echo
echo "prepared: $(date)"
echo

"$BIN" place \
    --tree "$TREE" \
    --root-seq "$ROOT" \
    --prepared work/prepared_2021.tsv \
    --seed "$SEED" \
    --threads "$THREADS" \
    --out-tree work/tree_2021.tsv \
    --out work/sets_2021.tsv \
    --out-examples work/examples_2021.txt \
    --out-matches work/matches_2021.tsv \
    --out-counts work/counts_2021.tsv \
    --out-msa work/msa_2021.fasta

echo
echo "placed : $(date)"
echo

mkdir -p "$OUTDIR"
cp work/examples_2021.txt "$OUTDIR/examples_2021_cur.txt"
cp work/tree_2021.tsv     "$OUTDIR/tree_2021_cur.tsv"
cp work/sets_2021.tsv     "$OUTDIR/sets_2021_cur.tsv"
cp work/matches_2021.tsv  "$OUTDIR/matches_2021_cur.tsv"
cp work/counts_2021.tsv   "$OUTDIR/counts_2021_cur.tsv"
cp work/msa_2021.fasta    "$OUTDIR/msa_2021_cur.fasta"
echo "copied : $OUTDIR"
echo

set +e
python3 -I check_examples.py work/examples_2021.txt
echo "checker: exit $?"
set -e

echo
echo "done   : $(date)"
