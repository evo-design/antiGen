#!/bin/bash
# The whole open-data queue -- no date cutoff -- placed on the 2026-03-15 all-time
# tree, with the BA.2 variant flags and the POST_CUTOFF flags. prepare and place in
# one job, for the reason run_2021.sbatch gives: the queue on disk was selected
# with the clock filter off, and Directive, 2026-10-22 (seventh) turns the envelope
# on for both runs.
#
# This is THE test tree, and there is only one: it holds every sequence, so it is
# a superset of every train arm's tree and can be labelled against every cutoff at
# once. --post-cutoff takes the whole of slurm/cutoffs.tsv -- the same file the
# train array is driven by, so the two cannot drift -- and each record lists the
# cutoffs every observed date below it is at or after. A train arm at cutoff D
# reads its test set off this one file by keeping the records whose POST_CUTOFF
# carries D. See Directive, 2026-10-23 (third).
#
# A batch job rather than a nohup for the reason NOTES records twice: a nohup'd run
# belongs to the session that started it and dies with it.
#
# 32 cores: the scaling curve peaks at 16 on a 75k-node tree and is flat to 32
# (53.4 s against 55.5 s), and this queue builds a tree four times that size, which
# is more scan per placement and so a higher parallel fraction. 64 is past the knee
# on every measurement so far.
#SBATCH --job-name=antigen-all
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=work/run_all.%j.log
#
# --exclude: the nodes whose /media/8TBNVME/scratch is not set up. They are the
# *idle* ones, which is why the scheduler prefers them and why this list is not
# optional -- see NOTES, "It is the idle nodes, and that is schedulable around".
# Drop it once the cluster is fixed; `sinfo -p cpu_batch_high_mem -N` and a look
# at which nodes jobs actually ran on is how the list was arrived at.
#SBATCH --exclude=GPU0F98,GPU115A,GPU1034,GPU104C

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BIN=${ANTIGEN_BIN:-work/antigen_all}
THREADS=${ANTIGEN_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
SEED=${ANTIGEN_SEED:-1}
TREE=${ANTIGEN_TREE:-data/ncov_open_global_all-time@2026-03-15.json}
ROOT=${ANTIGEN_ROOT:-data/ncov_open_global_all-time@2026-03-15_root-sequence.json}
FASTA=${ANTIGEN_FASTA:-$HOME/sc2-open-data/spike.fasta}
META=${ANTIGEN_META:-$HOME/sc2-open-data/metadata.tsv}
# One tree for every cutoff, so the directory is not any one cutoff's.
OUTDIR=${ANTIGEN_OUTDIR:-output/sc2-test}
CUTOFFS=${ANTIGEN_CUTOFFS:-$(cut -f1 slurm/cutoffs.tsv | paste -sd,)}

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none}"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "seed   : $SEED"
echo "threads: $THREADS (the tree written does not depend on this)"
echo "cutoffs: $CUTOFFS"
echo "start  : $(date)"
echo

"$BIN" prepare \
    --root-seq "$ROOT" \
    --fasta "$FASTA" \
    --metadata "$META" \
    --date-cutoff none \
    --out work/prepared_all.tsv \
    --out-clock work/clock_all.tsv

echo
echo "prepared: $(date)"
echo

# No --out-msa: over this queue it is ~11 GB, which is the prepared file again in
# another encoding, and that file is on disk beside it.
"$BIN" place \
    --tree "$TREE" \
    --root-seq "$ROOT" \
    --prepared work/prepared_all.tsv \
    --seed "$SEED" \
    --threads "$THREADS" \
    --variant BA.2 \
    --post-cutoff "$CUTOFFS" \
    --out-tree work/tree_all.tsv \
    --out work/sets_all.tsv \
    --out-examples work/examples_all.txt \
    --out-matches work/matches_all.tsv \
    --out-counts work/counts_all.tsv

echo
echo "placed : $(date)"
echo

mkdir -p "$OUTDIR"
cp work/examples_all.txt "$OUTDIR/examples_all_cur.txt"
cp work/tree_all.tsv     "$OUTDIR/tree_all_cur.tsv"
cp work/sets_all.tsv     "$OUTDIR/sets_all_cur.tsv"
cp work/counts_all.tsv   "$OUTDIR/counts_all_cur.tsv"
echo "copied : $OUTDIR (matches_all.tsv left in work/, it is 8.5M rows)"
echo

set +e
python3 -I check_examples.py work/examples_all.txt
echo "checker: exit $?"
set -e

echo
echo "done   : $(date)"
