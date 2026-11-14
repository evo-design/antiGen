#!/bin/bash
#SBATCH --job-name=denovo_diag
#SBATCH --partition=quick_preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/aakarsh/projects/antiGen/experiments/multitree/logs/denovo_diag_%j.out
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
$PY score_denovo_diag.py
