#!/bin/bash
#SBATCH --job-name=strat49
#SBATCH --partition=gpu_batch
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.err
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
"$PY" "$HERE/score_stratify_49tree.py"
echo "DONE scoring $(date)"
