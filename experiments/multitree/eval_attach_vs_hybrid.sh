#!/bin/bash
#SBATCH --job-name=attvshyb
#SBATCH --partition=gpu_batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%j.err
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
echo "START $(date) node $(hostname)"
"$PY" score_attach_vs_hybrid.py
echo "END $(date)"
