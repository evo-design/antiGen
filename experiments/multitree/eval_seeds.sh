#!/bin/bash
#SBATCH --job-name=j13eval
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
TREES="ncov h5nx measles rsv_a rsv_b denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha hmpv_b_g vic_ha"
# per-epoch val-vs-test for the two new seeds
$PY -I score_epochs.py "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s1" --trees $TREES --out "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s1/epochs_valtest.json"
$PY -I score_epochs.py "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s2" --trees $TREES --out "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s2/epochs_valtest.json"
# 3-seed ensemble (each seed's best-val checkpoint)
C0=$(ls "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s0"/*_best.pt | grep -v _best_ | head -1)
C1=$(ls "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s1"/*_best.pt | grep -v _best_ | head -1)
C2=$(ls "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s2"/*_best.pt | grep -v _best_ | head -1)
$PY -I ensemble_score.py --checkpoints "$C0" "$C1" "$C2" --trees $TREES     --out "/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/ensemble3_recall@2025-01-01.json"
echo "SEED EVAL DONE"
