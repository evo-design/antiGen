#!/bin/bash
#SBATCH --job-name=denovo_panels
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/denovo_panels/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/denovo_panels/logs/%x_%j.err

# Re-eval de-novo (+ all) recall for every Figure-5 panel from ONE 650M no-attach
# pipeline, with the canonical de-novo pool (training-seen cells struck). Writes a tidy
# CSV + per-event npz + provenance JSON. No hardcoded numbers; arms resolved at run time.
set -euo pipefail

PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SC=/home/aakarsh/projects/antiGen/experiments/multitree/score_denovo_panels.py
mkdir -p /large_storage/hielab/aakarsh/multitree/fc13_core_surface/denovo_panels/logs

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "git=$(git -C /home/aakarsh/projects/antiGen rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# run the interpreter directly (single task, single GPU); srun trips over a
# SLURM_CPUS_PER_TASK / SLURM_TRES_PER_TASK mismatch in this cluster's env.
"$PY" "$SC"
