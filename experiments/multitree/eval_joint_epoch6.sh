#!/bin/bash
#SBATCH --job-name=fcEp6
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SCORER=/home/aakarsh/projects/antiGen/experiments/multitree/score_forecast_recall.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
# Curiosity: the epoch-6 joint snapshot (= _best_denv1.pt, denv1's peak epoch) scored on
# ALL trees -- a single early-plateau checkpoint across every virus.
"$PY" "$SCORER" "/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0" --checkpoint "/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/fitness_esm2_esm2_t33_650M_UR50D_multitree14_randval0.1s0_lr0.0001_20261025_024216_best_denv1.pt" --data "$FDATA" --date 2025-01-01 \
    --trees h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha hmpv_a_g hmpv_b_g h9n2_ha vic_ha \
    --subset all novel --output "/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/recall_epoch6_alltrees@2025-01-01.json"
echo "EPOCH6 EVAL DONE"
