#!/bin/bash
#SBATCH --job-name=zs_probe
# One partition, not a list: listing several schedules the job against the most
# contended member (NOTES s15).
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --output=logs/zs_probe_%j.out
#SBATCH --error=logs/zs_probe_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/zeroshot_probe.py "$@"
