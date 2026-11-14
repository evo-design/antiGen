#!/bin/bash
#SBATCH --job-name=zeroshot_fig13
#SBATCH --partition=gpu_batch
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_eval/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_eval/logs/%x_%j.err

# Score the base ESM-2 650M zero-shot on the 13 no-attach test sets and merge a
# `zeroshot` rung into perevent_fig13_sp4e0.npz. GPU required (650M forward).
set -uo pipefail
mkdir -p /large_storage/hielab/aakarsh/multitree/fc13_eval/logs
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
"$PY" score_zeroshot_fig13.py
echo "DONE zeroshot $(date)"
