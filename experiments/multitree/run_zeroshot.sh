#!/bin/bash
#SBATCH --job-name=zeroshot
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/zeroshot_eval.py "$@"
