#!/bin/bash
#SBATCH --job-name=fig5_metrics
# `gpu` and `gpu_batch` are the SAME ten nodes -- they differ only in time
# limit (1 day vs 14) and in how contended the queue is. On 2026-10-14 an
# identical 1-GPU request was scheduled at 14:01 on `gpu` and 19:54 on
# `gpu_batch`, ~6 hours apart on the same hardware, so list `gpu` first and
# keep the others as fallback. Check with:
#   sbatch --test-only --partition=<P> --gpus=1 --mem=96G --wrap=true
#SBATCH --partition=gpu,gpu_batch,gpu_high_mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --output=logs/fig5_metrics_%j.out
#SBATCH --error=logs/fig5_metrics_%j.err
set -euo pipefail
cd /home/ivan.specht/antiGen
python experiments/multitree/collect_figure_metrics.py "$@"
