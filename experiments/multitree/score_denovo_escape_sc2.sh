#!/bin/bash
#SBATCH --job-name=esc_sc2
#SBATCH --partition=quick_preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aakarsh/projects/antiGen/experiments/multitree/logs/esc_sc2_%j.out
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
$PY score_denovo_escape_sc2.py
