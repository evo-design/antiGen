#!/bin/bash
# Fetch every row of groups.tsv with fetch_group.py, one at a time (NCBI allows ~3
# requests/s without a key; parallel fetches only earn 429s). NOTES s155.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/large_storage/hielab/ispecht/fitness-inference/corpus2023/viral}
PY=/home/ivan.specht/miniconda/bin/python
mkdir -p "$OUT"
grep -v '^#' "$HERE/groups.tsv" | while IFS=$'\t' read -r name slen query; do
    [ -n "${ONLY:-}" ] && ! printf '%s\n' $ONLY | grep -qx "$name" && continue
    [ -s "$OUT/$name/raw.faa" ] && { echo "$name: already fetched"; continue; }
    "$PY" -u "$HERE/fetch_group.py" --name "$name" --slen "$slen" --query "$query" \
        --out "$OUT" < /dev/null || echo "$name: FETCH FAILED"
done
