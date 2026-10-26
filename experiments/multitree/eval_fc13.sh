#!/bin/bash
#SBATCH --job-name=fc13eval
#SBATCH --partition=gpu_batch
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_eval/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_eval/logs/%x_%j.err

# Score one arm against one test-data dir with score_forecast_recall.py (recall@q for the
# model + count/where baselines, all + de novo). GPU required -- the 650M eval is far too
# slow on the login node.
#   $1 KIND    joint | scratch
#   $2 DATADIR absolute path to the data dir holding examples_<tree>@D_{train,test,novel}.txt
#   $3 TAG     short label for the output json (e.g. noattach, attachtest)
#   $4 RUN_DIR (joint only) the arm's output dir with *_best.pt
#   rest       trees to score
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SC=/home/aakarsh/projects/antiGen/experiments/multitree/score_forecast_recall.py
D=2025-01-01
KIND=$1; DATADIR=$2; TAG=$3; shift 3
mkdir -p /large_storage/hielab/aakarsh/multitree/fc13_eval/logs

if [ "$KIND" = joint ]; then
  RUN_DIR=$1; shift
  echo "JOINT eval  run=$RUN_DIR  data=$DATADIR  tag=$TAG  trees=$*"
  "$PY" "$SC" "$RUN_DIR" --data "$DATADIR" --date "$D" --trees "$@" \
      --subset all novel --output "$RUN_DIR/scores_recall_${TAG}.json"
elif [ "$KIND" = scratch ]; then
  ROOT=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer
  for t in "$@"; do
    rd="$ROOT/single_${t}_s0"
    [ -d "$rd" ] || { echo "skip $t (no run dir)"; continue; }
    echo "SCRATCH eval  tree=$t  data=$DATADIR  tag=$TAG"
    "$PY" "$SC" "$rd" --data "$DATADIR" --date "$D" --trees "$t" \
        --subset all novel --output "$rd/scores_recall_${TAG}.json" || echo "  FAILED $t"
  done
else echo "unknown KIND $KIND" >&2; exit 1; fi
echo "DONE $(date)"
