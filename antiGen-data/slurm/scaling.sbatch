#!/bin/bash
# Thread scaling for the placement scan, on one node with many cores.
#
# Runs the same queue at several thread counts and checks, as it goes, that the
# tree written is byte-identical at every one of them -- the property the
# parallel scan is built around. Speed is the point of the change; sameness is
# the thing that makes it safe.
#SBATCH --job-name=antigen-scaling
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=work/scaling.%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BIN=${ANTIGEN_BIN:-work/antigen_scaling}
PREPARED=${ANTIGEN_PREPARED:?set ANTIGEN_PREPARED to the queue to place}
TREE=${ANTIGEN_TREE:-data/ncov_open_global_all-time@2026-03-15.json}
ROOT=${ANTIGEN_ROOT:-data/ncov_open_global_all-time@2026-03-15_root-sequence.json}
COUNTS=${ANTIGEN_THREADS_LIST:-"1 2 4 8 16 32 64"}
OUT=${ANTIGEN_SCRATCH:-work/scaling.$$}

mkdir -p "$OUT"
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "cores   : ${SLURM_CPUS_PER_TASK:-?}"
echo "binary  : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "queue   : $PREPARED"
echo "counts  : $COUNTS"
echo

base=""
for th in $COUNTS; do
    start=$(date +%s.%N)
    "$BIN" place \
        --tree "$TREE" --root-seq "$ROOT" --prepared "$PREPARED" \
        --seed 1 --threads "$th" \
        --out-tree "$OUT/tree.$th.tsv" > "$OUT/run.$th.log" 2>&1
    end=$(date +%s.%N)
    secs=$(echo "$end - $start" | bc)
    sum=$(md5sum < "$OUT/tree.$th.tsv" | cut -d' ' -f1)
    [ -z "$base" ] && base="$sum"
    same=$([ "$sum" = "$base" ] && echo same || echo DIFFERENT)
    placed=$(grep -o 'in [0-9.]* s, tree now [0-9]*' "$OUT/run.$th.log" | tail -1)
    printf 'threads %3s : %8.1f s   tree %s   %s\n' "$th" "$secs" "$same" "$placed"
    rm -f "$OUT/tree.$th.tsv"
done

echo
echo "done    : $(date)"
