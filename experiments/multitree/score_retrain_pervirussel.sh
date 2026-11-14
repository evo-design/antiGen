#!/bin/bash
#SBATCH --job-name=fc13pvsel
#SBATCH --partition=gpu_batch
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.err
set -euo pipefail
/home/aakarsh/miniconda/envs/ag_env/bin/python /home/aakarsh/projects/antiGen/experiments/multitree/score_retrain_pervirussel.py
