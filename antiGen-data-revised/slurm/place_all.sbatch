#!/bin/bash
# The whole open-data queue -- no date cutoff, no clock filter -- placed on the
# 2026-03-15 all-time tree, with the BA.2 variant flags and one POST_CUTOFF flag
# per quarterly cutoff, taken from slurm/cutoffs.tsv. This is the place half of
# slurm/run_all.sbatch, for a queue already prepared on disk.
#
# It is a batch job rather than a nohup for the reason NOTES records twice: a
# nohup'd run belongs to the session that started it and dies with it. The first
# attempt at this one died at 490,000 of 8,503,491 exactly that way.
#SBATCH --job-name=antigen-place-all
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=work/place_all.%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BIN=${ANTIGEN_BIN:-work/antigen_all}
THREADS=${ANTIGEN_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
SEED=${ANTIGEN_SEED:-1}
TREE=${ANTIGEN_TREE:-data/ncov_open_global_all-time@2026-03-15.json}
ROOT=${ANTIGEN_ROOT:-data/ncov_open_global_all-time@2026-03-15_root-sequence.json}
CUTOFFS=${ANTIGEN_CUTOFFS:-$(cut -f1 slurm/cutoffs.tsv | paste -sd,)}

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none}"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "seed   : $SEED"
echo "threads: $THREADS (the tree written does not depend on this)"
echo "start  : $(date)"
echo

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
echo "done   : $(date)"
