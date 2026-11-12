#!/bin/bash
# Build coronavirus backbones on the current allocation, two at a time x 8 threads.
OUT=${OUT:-/large_storage/hielab/ispecht/fitness-inference/corpus2023/cov}
PY=/home/ivan.specht/miniconda/bin/python
cd "$(dirname "$0")"
mkdir -p $OUT/logs
build() { g=$1; ref=""; [ "$g" = sarbeco ] || [ "$g" = sarbecovirus ] && ref="--reference sc2"
  $PY -u build_cov_tree.py $OUT/$g $ref --threads 8 --max-tips 1500 > $OUT/logs/${g}_local.log 2>&1 && echo "$g OK" || echo "$g FAILED"; }
export -f build; export OUT PY
printf '%s\n' ${GROUPS_:-sarbeco hku1 mhv nl63 hcov229e deltacov alphacov1 ibv betacov1 merbeco pedv} | xargs -P 2 -I{} bash -c 'build {}'
