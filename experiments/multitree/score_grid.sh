#!/bin/bash
#SBATCH --partition=gpu_batch
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_grid/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_grid/logs/%x_%j.err
set -euo pipefail
: "${SIZE:?}"; : "${FT:?}"
/home/aakarsh/miniconda/envs/ag_env/bin/python /home/aakarsh/projects/antiGen/experiments/multitree/score_grid.py --size "$SIZE" --ft "$FT"
