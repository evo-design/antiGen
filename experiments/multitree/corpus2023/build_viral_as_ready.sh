#!/bin/bash
# Build each groups.tsv group once its fetch has finished, two at a time x 8 threads.
# The E groups that polyprotein peptides merge into wait for fetch_polys.sh. NOTES s155.
HERE=$(cd "$(dirname "$0")" && pwd)
V=/large_storage/hielab/ispecht/fitness-inference/corpus2023/viral
L=/large_storage/hielab/ispecht/fitness-inference/corpus2023/viral_fetch.log
PL=/large_storage/hielab/ispecht/fitness-inference/corpus2023/poly_fetch.log
PY=/home/ivan.specht/miniconda/bin/python
mkdir -p $V/logs
POLY_TARGETS=" denv1_e denv2_e denv3_e denv4_e zika_e wnv_e chikv_e1 "
mapfile -t ALL < <(grep -v '^#' $HERE/groups.tsv | cut -f1)
declare -A started
while :; do
    pending=0
    for g in "${ALL[@]}"; do
        [ -n "${started[$g]:-}" ] && continue
        pending=1
        ready=0
        grep -q "^$g: wrote" $L 2>/dev/null && ready=1
        [ "$g" = mumps_hn ] && { grep -c "^mumps_hn: wrote" $L | grep -q 2 && ready=1 || ready=0; }
        if [[ "$POLY_TARGETS" == *" $g "* ]]; then
            grep -q "added to $V/$g\$" $PL 2>/dev/null || ready=0
        fi
        [ $ready = 1 ] || continue
        while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 20; done
        started[$g]=1
        ( $PY -u $HERE/build_cov_tree.py $V/$g --threads 8 --max-tips 1500 > $V/logs/$g.log 2>&1 \
            && echo "$g OK" || echo "$g FAILED" ) &
    done
    [ $pending = 0 ] && break
    sleep 60
done
wait
