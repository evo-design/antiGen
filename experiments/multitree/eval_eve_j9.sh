#!/bin/bash
#SBATCH --job-name=eve_j9
#SBATCH --partition=gpu,gpu_high_mem,gpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-8
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%A_%a.err
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/aakarsh/projects/antiGen/experiments/multitree
echo "START $(date) node $(hostname) task ${SLURM_ARRAY_TASK_ID}"
/home/aakarsh/miniconda/envs/ag_env/bin/python score_eve_j9.py "${SLURM_ARRAY_TASK_ID}"
echo "END $(date)"
