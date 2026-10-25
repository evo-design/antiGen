#!/bin/bash
#SBATCH --job-name=fcPerEv
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/aakarsh/miniconda/envs/ag_env/bin/python -I /home/aakarsh/projects/antiGen/experiments/multitree/score_perevent.py
