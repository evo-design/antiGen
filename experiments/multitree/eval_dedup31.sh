#!/bin/bash
#SBATCH --job-name=evdedup31
#SBATCH --partition=gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_dedup31/logs/evdedup31_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_dedup31/logs/evdedup31_%j.err
set -euo pipefail
cd /home/aakarsh/projects/antiGen/experiments/multitree
echo "START $(date)"
/home/aakarsh/miniconda/envs/ag_env/bin/python score_dedup31.py
echo "END $(date)"
