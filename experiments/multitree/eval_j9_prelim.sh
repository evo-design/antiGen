#!/bin/bash
#SBATCH --job-name=eval_j9p
#SBATCH --partition=gpu_batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/eval_j9p_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/eval_j9p_%j.err

# PRELIMINARY 9-virus eval: joint9/joint49/xformer at 10 seeds, single-650p4 at whatever
# clean seeds are currently available per tree (the 5 corrupt runs are retraining). Writes
# to multitree_noattach_j9_prelim/ so it never collides with the final 10-seed eval.
set -euo pipefail
export J9_SUFFIX=_prelim
# skip the 5 runs currently retraining, to avoid reading checkpoints mid-write
export J9_SINGLE_EXCLUDE="h5nx:4,6,7,8;h3n2_ha:8"
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
cd "$HERE"
echo "START $(date) node $(hostname)  J9_SUFFIX=$J9_SUFFIX"
"$PY" score_HI_j9.py
"$PY" score_xformer_j9.py
"$PY" build_j9_macro.py
echo "END $(date)"
