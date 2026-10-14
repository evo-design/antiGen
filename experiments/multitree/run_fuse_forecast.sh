#!/bin/bash
#SBATCH --job-name=fuse_fcst
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --output=logs/fuse_fcst_%j.out
#SBATCH --error=logs/fuse_fcst_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/fuse_forecast.py "$@"
