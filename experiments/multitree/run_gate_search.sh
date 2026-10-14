#!/bin/bash
#SBATCH --job-name=gate_search
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/gate_search_%j.out
#SBATCH --error=logs/gate_search_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/gate_search.py "$@"
