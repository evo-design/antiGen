#!/bin/bash
#SBATCH --job-name=nll_audit
# One partition, not a list: listing several schedules the job against the most
# contended member (NOTES s15).
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --output=logs/nll_audit_%j.out
#SBATCH --error=logs/nll_audit_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/figure_nll_audit.py "$@"
