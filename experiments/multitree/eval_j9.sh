#!/bin/bash
#SBATCH --job-name=eval_j9
#SBATCH --partition=gpu_batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/eval_j9_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/eval_j9_%j.err

# 9-virus / 10-seed eval for the new panel: joint9, joint49, single-650p4, xformer
# (joint13 dropped). Produces per-event npz + per-virus + pooled ladder + forest,
# then the macro recall@10% table. Meant to run after joint49 seed 8 completes
# (submit with --dependency=afterany:<seed8 jobid>).
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
cd "$HERE"
echo "START $(date) node $(hostname)"
echo "=== score_HI_j9 (single/joint9/joint49, 10-seed MeanLogProb, posthoc NLL) ==="
"$PY" score_HI_j9.py
echo "=== score_xformer_j9 (from-scratch, 10-seed) ==="
"$PY" score_xformer_j9.py
echo "=== build_j9_macro (macro table) ==="
"$PY" build_j9_macro.py
echo "END $(date)"
