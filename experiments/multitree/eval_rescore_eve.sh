#!/bin/bash
#SBATCH --job-name=eve_rescore
#SBATCH --partition=gpu,gpu_high_mem,gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%j.err
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
echo "START $(date) node $(hostname)"
"$PY" rescore_eve_j9.py
echo "END $(date)"
