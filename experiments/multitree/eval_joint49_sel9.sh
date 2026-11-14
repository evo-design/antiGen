#!/bin/bash
#SBATCH --job-name=j49sel9
#SBATCH --partition=gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/j49sel9_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/j49sel9_%j.err
set -euo pipefail
cd /home/aakarsh/projects/antiGen/experiments/multitree
echo "START $(date)"
/home/aakarsh/miniconda/envs/ag_env/bin/python score_joint49_sel9.py
echo "END $(date)"
