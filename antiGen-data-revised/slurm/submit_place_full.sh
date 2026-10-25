#!/bin/bash
# Snapshot the binary, then submit. The snapshot is what the job runs, so the
# working tree can be rebuilt freely while it is queued or running.
set -euo pipefail
cd "$(dirname "$0")/.."
make
cp antigen work/antigen_full
sbatch "$@" slurm/place_full.sbatch
