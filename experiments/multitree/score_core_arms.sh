#!/bin/bash
#SBATCH --job-name=fc13corescore
#SBATCH --partition=gpu_batch
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.err

# Score the 4 core-surface joint arms on the 13-panel no-attach eval with the fig13
# metric (per-event hit@10, macro, all + de novo) at epoch 3. GPU required (650M eval is
# far too slow on the login node). Writes the per-event NPZ + ladder_core_arms.csv.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
SC=$HERE/score_core_arms_fig13.py
R=/large_storage/hielab/aakarsh/multitree/fc13_core_surface
# These arms train 3 epochs saved 0-indexed (epoch0/1/2), so the final/reporting epoch
# is 2 -- NOT 3.
EPOCH=${EPOCH:-2}

# Apples-to-apples baseline: the 13-panel-only joint, SAME recipe as these arms
# (LM-only, no val split, 3 epochs, epoch2 final). The published fig13 `joint` rung is a
# 5-seed val-selected ensemble and sits a little higher; this is the fair comparison.
J13=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/joint13_lmonly_noattach_noval

"$PY" "$SC" --epoch "$EPOCH" \
    --arm baseline_joint13_s0=$J13 \
    --arm joint_surface_snapshot_partial4=$R/joint_surface_snapshot_partial4_noattach_noval \
    --arm joint_surface_all_partial4=$R/joint_surface_all_partial4_noattach_noval \
    --arm joint_surface_snapshot_lmonly=$R/joint_surface_snapshot_lmonly_noattach_noval \
    --arm joint_surface_all_lmonly=$R/joint_surface_all_lmonly_noattach_noval \
    --out-npz $R/perevent_core_arms.npz
echo "DONE $(date)"
