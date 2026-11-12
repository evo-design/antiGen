#!/bin/bash
# Polyprotein records whose envelope peptide is cut out and merged into the matching
# protein group (extract_matpeptide.py). NOTES s155.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
V=${OUT:-/large_storage/hielab/ispecht/fitness-inference/corpus2023/viral}
PY=/home/ivan.specht/miniconda/bin/python
grep -v '^#' "$HERE/polys.tsv" | while IFS=$'\t' read -r name slen prod into query; do
    [ -s "$V/_poly/$name/raw.faa" ] || "$PY" -u "$HERE/fetch_group.py" --name "$name" \
        --slen "$slen" --query "$query" --out "$V/_poly" < /dev/null || { echo "$name: FETCH FAILED"; continue; }
    mkdir -p "$V/$into"
    "$PY" -u "$HERE/extract_matpeptide.py" "$V/_poly/$name" --product "$prod" --merge-into "$V/$into" < /dev/null
done
