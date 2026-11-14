#!/bin/bash
#SBATCH --job-name=fig4score
#SBATCH --partition=gpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fig4_single_attach/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fig4_single_attach/logs/%x_%j.err
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
$PY score_fig4_panels.py
