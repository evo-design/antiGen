#!/bin/bash
#SBATCH --job-name=eval_wide
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/eval_wide_%j.out
#SBATCH --error=logs/eval_wide_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/eval_wide.py "$@"
