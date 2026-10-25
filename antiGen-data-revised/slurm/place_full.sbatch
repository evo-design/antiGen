#!/bin/bash
# The full placement run: every prepared sequence, then the tree, the allele sets,
# the examples file and the provenance of the sequences that matched a node and
# were not attached. Single-threaded and a couple of minutes since a match is no
# longer attached (it was 3.9 h), but it must still outlive the interactive
# allocation it is submitted from -- an earlier attempt was run with nohup inside
# one and died with it at 190k of 351k placements.
#
# Submit with slurm/submit_place_full.sh, which snapshots the binary first so a
# rebuild in the working tree cannot swap it out mid-run.
#SBATCH --job-name=antigen-place
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=work/place_full.%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BIN=${ANTIGEN_BIN:-work/antigen_full}
SEED=${ANTIGEN_SEED:-1}

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none}"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "seed   : $SEED"
echo

"$BIN" place \
    --tree data/ncov_global@2020-12-29.json \
    --root-seq data/ncov_global@2020-12-29_root-sequence.json \
    --prepared work/prepared_2021.tsv \
    --seed "$SEED" \
    --out-tree work/tree_2021.tsv \
    --out work/sets_2021.tsv \
    --out-examples work/examples_2021.txt \
    --out-matches work/matches_2021.tsv \
    --out-counts work/counts_2021.tsv

echo
echo "done   : $(date)"
