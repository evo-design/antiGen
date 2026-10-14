#!/bin/bash
#SBATCH --job-name=fig5_metrics
#SBATCH --partition=gpu_batch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --output=logs/fig5_metrics_%j.out
#SBATCH --error=logs/fig5_metrics_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/collect_figure_metrics.py "$@"
