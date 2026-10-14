#!/bin/bash
#SBATCH --job-name=fc_precision
#SBATCH --partition=gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/precision_%j.out
#SBATCH --error=logs/precision_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/forecast_precision.py "$@"
