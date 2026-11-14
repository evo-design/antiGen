#!/bin/bash
#SBATCH --job-name=j13_fig13
#SBATCH --partition=quick_preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/aakarsh/projects/antiGen/experiments/multitree/logs/j13_fig13_%j.out
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
R=/large_storage/hielab/aakarsh/multitree
# sp4e0 = surface partial-4, epoch 0. joint (49) = multitree49 snapshot; joint13 = multitree13.
$PY score_core_arms_fig13.py --epoch 0 \
  --arm joint_verify=$R/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval \
  --arm joint13=$R/fc13_joint_650m/joint13_partial4_noattach_noval \
  --out-npz $R/fc13_core_surface/perevent_joint13_sp4e0.npz \
  --ladder /home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/ladder_joint13_check.csv
