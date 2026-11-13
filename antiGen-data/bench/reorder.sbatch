#!/bin/bash
#SBATCH --job-name=ubench-order
#SBATCH --partition=cpu_high_mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/large_storage/hielab/ivan.specht/usher-bench/logs/order_%j.out
#
# The placement order for each N (bench/reorder_inputs.py), into inputs_chrono/.
#   sbatch bench/reorder.sbatch 1000 3000 10000 ...      plain
#   sbatch bench/reorder.sbatch --batch 300000 1000000   also cut 100k batches
set -euo pipefail
PY=/home/ivan.specht/miniconda/bin/python3
BD=/large_storage/hielab/ivan.specht/usher-bench
BATCH=()
if [ "$1" = --batch ]; then BATCH=(--batch 100000); shift; fi
for N in "$@"; do
  echo "[$N] $(date)"
  $PY -I "${SLURM_SUBMIT_DIR:-$PWD}"/bench/reorder_inputs.py "$BD/inputs" "$BD/inputs_chrono" "$N" "${BATCH[@]}"
done
echo "done $(date)"
