#!/bin/bash
# Build every coronavirus spike backbone (build_cov_tree.py), one cpu_batch task per
# group (CLAUDE.md: batch CPU work goes to cpu_batch, never cpu). NOTES s155.
#
# The sarbecoviruses are built in the SARS-CoV-2 frame (--reference sc2); every other
# group in its own RefSeq (or modal-length) frame.
set -euo pipefail
OUT=${OUT:-/large_storage/hielab/ispecht/fitness-inference/corpus2023/cov}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/ivan.specht/miniconda/bin/python
GROUPS_=${ONLY:-"sarbeco merbeco betacov1 hku1 mhv hcov229e nl63 pedv alphacov1 ibv deltacov"}
mkdir -p "$OUT/logs"
for g in $GROUPS_; do
    [ -s "$OUT/$g/raw.faa" ] || { echo "skip $g: no raw.faa"; continue; }
    ref=""; [ "$g" = sarbeco ] && ref="--reference sc2"
    sbatch --job-name=covtree_$g --partition=cpu_batch --time=12:00:00 \
        --cpus-per-task=8 --mem=32G \
        --output="$OUT/logs/${g}_%j.out" \
        --wrap "cd $HERE && $PY -u build_cov_tree.py $OUT/$g $ref --threads 8 --max-tips ${MAX_TIPS:-1500}"
done
