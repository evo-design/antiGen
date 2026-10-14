#!/bin/bash
#SBATCH --job-name=seed_curve
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/seed_curve_%j.out
#SBATCH --error=logs/seed_curve_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/seed_curve.py "$@"
